#include "hand_core.h"
#include "hagrid.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <set>
#include <tuple>

namespace hand {
static double clip(double x,double lo=0,double hi=1) { return std::max(lo,std::min(hi,x)); }
static double median(std::vector<double> v) {
    const size_t n=v.size(); if(!n) throw std::runtime_error("empty median");
    std::nth_element(v.begin(),v.begin()+n/2,v.end()); double m=v[n/2];
    if(n%2==0) m=(m+*std::max_element(v.begin(),v.begin()+n/2))*.5;
    return m;
}
cv::Mat palmInput(const cv::Mat& rgb,cv::Point2d& bias) {
    double ratio=192./std::max(rgb.rows,rgb.cols);
    cv::Size size(int(rgb.cols*ratio),int(rgb.rows*ratio));
    if(size.width<1||size.height<1) throw std::runtime_error("palm input size");
    cv::Mat resized,padded; cv::resize(rgb,resized,size);
    int left=(192-size.width)/2,top=(192-size.height)/2;
    cv::copyMakeBorder(resized,padded,top,192-size.height-top,left,192-size.width-left,cv::BORDER_CONSTANT,cv::Scalar());
    bias={double(int(left/ratio)),double(int(top/ratio))};
    return padded;
}
std::vector<Palm> decodePalms(const Outputs& o,cv::Size size,cv::Point2d bias,double threshold,double nms) {
    if(o.size()!=2||o[0].size()!=2016*18||o[1].size()!=2016) throw std::runtime_error("palm output shape");
    std::vector<Palm> all; int idx=0; double scale=std::max(size.width,size.height);
    for(int grid: {24,12}) for(int y=0;y<grid;y++) for(int x=0;x<grid;x++)
        for(int a=0;a<(grid==24?2:6);a++,idx++) {
            double score=1/(1+std::exp(-clip(o[1][idx],-80,80))); if(score<threshold) continue;
            const float* v=o[0].data()+idx*18; double ax=(x+.5)/grid,ay=(y+.5)/grid;
            Palm p; p.score=score;
            p.box={(ax+(v[0]-.5*v[2])/192)*scale-bias.x,(ay+(v[1]-.5*v[3])/192)*scale-bias.y,
                   (ax+(v[0]+.5*v[2])/192)*scale-bias.x,(ay+(v[1]+.5*v[3])/192)*scale-bias.y};
            if(p.box[2]<=p.box[0]||p.box[3]<=p.box[1]) continue;
            for(int k=0;k<7;k++) p.points[k]={(ax+v[4+k*2]/192)*scale-bias.x,(ay+v[5+k*2]/192)*scale-bias.y};
            all.push_back(p);
        }
    std::stable_sort(all.begin(),all.end(),[](const Palm&a,const Palm&b){return a.score>b.score;});
    std::vector<Palm> result;
    for(const Palm& p:all) {
        bool keep=true;
        for(const Palm& q:result) {
            double inter=std::max(0.,std::min(p.box[2],q.box[2])-std::max(p.box[0],q.box[0]))*
                         std::max(0.,std::min(p.box[3],q.box[3])-std::max(p.box[1],q.box[1]));
            double area=(p.box[2]-p.box[0])*(p.box[3]-p.box[1])+(q.box[2]-q.box[0])*(q.box[3]-q.box[1])-inter;
            if(area>0&&inter/area>nms) {keep=false;break;}
        }
        if(keep) result.push_back(p);
        if(result.size()==2) break;
    }
    return result;
}
static cv::Mat cropPad(const cv::Mat& image,cv::Vec4d& box,cv::Point2d& bias,bool rotate) {
    double w=box[2]-box[0],h=box[3]-box[1],cx=(box[0]+box[2])*.5,cy=(box[1]+box[3])*.5;
    if(!rotate) cy-=.4*h;
    double factor=rotate?4.:3.;
    int x1=int(clip(int(cx-w*factor*.5),0,image.cols)),x2=int(clip(int(cx+w*factor*.5),0,image.cols));
    int y1=int(clip(int(cy-h*factor*.5),0,image.rows)),y2=int(clip(int(cy+h*factor*.5),0,image.rows));
    if(x2-x1<2||y2-y1<2) return {};
    box={double(x1),double(y1),double(x2),double(y2)};
    int side=rotate?int(std::hypot(x2-x1,y2-y1)):std::max(x2-x1,y2-y1);
    int left=(side-(x2-x1))/2,top=(side-(y2-y1))/2;
    cv::Mat padded;cv::copyMakeBorder(image(cv::Rect(x1,y1,x2-x1,y2-y1)),padded,top,side-(y2-y1)-top,left,side-(x2-x1)-left,cv::BORDER_CONSTANT|cv::BORDER_ISOLATED,cv::Scalar());
    bias={double(x1-left),double(y1-top)};
    return padded;
}
Crop poseInput(const cv::Mat& rgb,const Palm& palm) {
    Crop c; c.box=palm.box;
    cv::Mat image=cropPad(rgb,c.box,c.bias,true); if(image.empty()) return c;
    cv::Point2d p0=palm.points[0]-c.bias,p2=palm.points[2]-c.bias;
    double radians=CV_PI/2-std::atan2(-(p2.y-p0.y),p2.x-p0.x);
    radians-=2*CV_PI*std::floor((radians+CV_PI)/(2*CV_PI));c.angle=radians*180/CV_PI;
    cv::Point2f center(float((c.box[0]+c.box[2])*.5-c.bias.x),float((c.box[1]+c.box[3])*.5-c.bias.y));
    c.rotation=cv::getRotationMatrix2D(center,c.angle,1.);
    cv::Mat rotated;cv::warpAffine(image,rotated,c.rotation,image.size());
    cv::Vec4d bounds(1e20,1e20,-1e20,-1e20);
    for(auto p:palm.points) {
        p-=c.bias;
        double x=c.rotation.at<double>(0,0)*p.x+c.rotation.at<double>(0,1)*p.y+c.rotation.at<double>(0,2);
        double y=c.rotation.at<double>(1,0)*p.x+c.rotation.at<double>(1,1)*p.y+c.rotation.at<double>(1,2);
        bounds[0]=std::min(bounds[0],x);bounds[1]=std::min(bounds[1],y);bounds[2]=std::max(bounds[2],x);bounds[3]=std::max(bounds[3],y);
    }
    cv::Point2d unused;cv::Mat crop=cropPad(rotated,bounds,unused,false);
    c.box=bounds;if(crop.empty()) return c;
    cv::resize(crop,c.input,{224,224},0,0,cv::INTER_AREA);return c;
}
bool decodePose(const Outputs& o,const Crop& c,double palmScore,Pose& p) {
    if(o.size()!=4||o[0].size()!=63||o[1].size()!=1||o[2].size()!=1||o[3].size()!=63) throw std::runtime_error("pose output shape");
    if(!std::isfinite(o[1][0])||o[1][0]<.7) return false;
    double scale=std::max(c.box[2]-c.box[0],c.box[3]-c.box[1])/224.;
    double co=std::cos(c.angle*CV_PI/180),si=std::sin(c.angle*CV_PI/180);
    cv::Mat inverse;cv::invertAffineTransform(c.rotation,inverse);
    double cx=(c.box[0]+c.box[2])*.5,cy=(c.box[1]+c.box[3])*.5;
    cv::Point2d origin(inverse.at<double>(0,0)*cx+inverse.at<double>(0,1)*cy+inverse.at<double>(0,2)+c.bias.x,
                       inverse.at<double>(1,0)*cx+inverse.at<double>(1,1)*cy+inverse.at<double>(1,2)+c.bias.y);
    for(int k=0;k<21;k++) {
        double x=(o[0][k*3]-112)*scale,y=(o[0][k*3+1]-112)*scale;
        p.points[k]={x*co-y*si+origin.x,x*si+y*co+origin.y,o[0][k*3+2]*scale};
        x=o[3][k*3];y=o[3][k*3+1];p.world[k]={x*co-y*si,x*si+y*co,o[3][k*3+2]};
        for(int j=0;j<3;j++) if(!std::isfinite(p.points[k][j])||!std::isfinite(p.world[k][j])) return false;
    }
    p.side=o[2][0]>.5;p.confidence=std::min(palmScore,double(o[1][0]));return true;
}
static double angle(const cv::Vec3d& a,const cv::Vec3d& b,const cv::Vec3d& c) {
    auto u=a-b,v=c-b;double d=cv::norm(u)*cv::norm(v);
    return std::acos(clip(u.dot(v)/d,-1,1))*180/CV_PI;
}
static bool scores(Joints p,cv::Vec3d& result) {
    double width=std::max(cv::norm(p[5]-p[17]),.7*cv::norm(p[9]-p[0]));if(width<1e-8) return false;
    auto origin=p[0];for(auto& v:p) v=(v-origin)/width;
    for(int k=0;k<21;k++) for(int j=0;j<3;j++) if(!std::isfinite(p[k][j])) return false;
    for(int k=0;k<4;k++) if(cv::norm(p[k+1]-p[k])<1e-8) return false;
    double opened=0,closed=0,flexion=0,pinch=1e20,opposition=1e20;
    for(int m:{5,9,13,17}) {
        double length=0;for(int j=m;j<m+3;j++) {double v=cv::norm(p[j+1]-p[j]);if(v<1e-8)return false;length+=v;}
        double a=angle(p[0],p[m],p[m+1]),b=angle(p[m],p[m+1],p[m+2]),c=angle(p[m+1],p[m+2],p[m+3]);
        double extension=cv::norm(p[m+3]-p[m])/length;
        opened+=.5*clip((a-90)/70)+.3*clip((b-70)/100)+.2*clip((c-70)/100);
        closed+=.55*clip((155-b)/80)+.25*clip((.95-extension)/.5)+.20*clip((150-a)/70);
        flexion+=clip((155-a)/65);
        double distance=cv::norm(p[4]-p[m+3]);opposition=std::min(opposition,distance);if(m<13)pinch=std::min(pinch,distance);
    }
    double grasp=.5*flexion/4+.5*clip((1.5-pinch)/1.2);grasp=std::max(grasp,.95*clip((.75-pinch)/.5));
    result={.8*opened/4+.2*clip((opposition-.35)/1.),grasp,closed/4};return true;
}
bool classify(Pose& p,cv::Vec3d* output) {
    cv::Vec3d a,b,total;bool aw=scores(p.world,a),bw=scores(p.points,b);if(!aw&&!bw)return false;
    total=(aw?a*.65:cv::Vec3d())+(bw?b*.35:cv::Vec3d());total/=((aw?.65:0)+(bw?.35:0));
    p.state=0;for(int j=1;j<3;j++)if(total[j]>total[p.state])p.state=j;if(output)*output=total;return true;
}
bool measure(Pose& p,const cv::Mat& depth,const cv::Matx33d& K,const cv::Vec<double,8>& D,bool matched) {
    p.measured=false;p.reference={double((float(p.points[0][0])+float(p.points[9][0]))*.5f),double((float(p.points[0][1])+float(p.points[9][1]))*.5f)};
    if(!matched||depth.empty()||depth.type()!=CV_16UC1||p.reference.x<0||p.reference.y<0||p.reference.x>=depth.cols||p.reference.y>=depth.rows)return false;
    if(!std::isfinite(K(0,0))||!std::isfinite(K(1,1))||K(0,0)<=0||K(1,1)<=0)return false;
    std::vector<cv::Point2f> points,hull;for(int i:{0,1,5,9,13,17})points.emplace_back(float(p.points[i][0]),float(p.points[i][1]));
    cv::convexHull(points,hull);if(hull.size()<3)return false;cv::Point2f center;
    for(auto v:hull)center+=v;center*=1.f/hull.size();
    std::vector<cv::Point> polygon;for(auto v:hull){v=center+(v-center)*.7f;polygon.emplace_back(cvRound(v.x),cvRound(v.y));}
    cv::Mat mask=cv::Mat::zeros(depth.size(),CV_8U);cv::fillConvexPoly(mask,polygon,cv::Scalar(1));cv::erode(mask,mask,cv::Mat::ones(3,3,CV_8U));
    int area=cv::countNonZero(mask);if(area<25)return false;
    std::vector<double> values;std::vector<cv::Point> pixels;
    for(int y=0;y<depth.rows;y++){const auto* d=depth.ptr<uint16_t>(y);const auto* m=mask.ptr<uint8_t>(y);
        for(int x=0;x<depth.cols;x++)if(m[x]&&d[x]>50&&d[x]<5000){values.push_back(d[x]);pixels.emplace_back(x,y);}}
    if(values.size()<12)return false;double middle=median(values);std::vector<double> deviation;
    for(double v:values)deviation.push_back(std::abs(v-middle));double cutoff=clip(3*1.4826*median(deviation),20,80);
    std::vector<double> kept;std::set<int> quadrants;
    for(size_t i=0;i<values.size();i++)if(std::abs(values[i]-middle)<=cutoff){kept.push_back(values[i]);quadrants.insert((pixels[i].x>=center.x)+2*(pixels[i].y>=center.y));}
    if(kept.size()<12||kept.size()<.6*values.size()||kept.size()<.01*area||quadrants.size()<3)return false;
    std::vector<cv::Point2d> ray;cv::undistortPoints(std::vector<cv::Point2d>{p.reference},ray,cv::Mat(K),cv::Mat(D));
    double z=median(kept);p.xyz={ray[0].x*z,ray[0].y*z,z};p.measured=true;return true;
}
Palm trackedPalm(const Pose& p) {
    Palm palm;palm.box={1e20,1e20,-1e20,-1e20};palm.score=p.confidence;int i=0;
    for(int j:{0,5,9,13,17,1,2}){cv::Point2d v(p.points[j][0],p.points[j][1]);palm.points[i++]=v;
        palm.box[0]=std::min(palm.box[0],v.x);palm.box[1]=std::min(palm.box[1],v.y);palm.box[2]=std::max(palm.box[2],v.x);palm.box[3]=std::max(palm.box[3],v.y);}
    return palm;
}
const char* label(int state){return gestureName(state);}
std::vector<Track> Tracker::update(std::vector<Pose>& poses,double now,cv::Size size) {
    using Match=std::tuple<double,double,int,int>;std::vector<Match> matches;
    double diagonal=std::hypot(size.width,size.height);std::set<int> available,used;std::vector<int> assignment(poses.size(),-1);
    for(size_t j=0;j<tracks.size();j++){available.insert(j);for(size_t i=0;i<poses.size();i++){
        double d=cv::norm(poses[i].reference-tracks[j].pose.reference);matches.emplace_back(d+(poses[i].side!=tracks[j].pose.side?.05*diagonal:0),d,int(i),int(j));}}
    std::sort(matches.begin(),matches.end());for(auto m:matches){int i=std::get<2>(m),j=std::get<3>(m);
        if(assignment[i]<0&&available.count(j)&&std::get<1>(m)<.2*diagonal){assignment[i]=j;available.erase(j);}}
    for(size_t i=0;i<poses.size();i++){
        Pose& p=poses[i];int j=assignment[i];if(j<0){Track t;t.pose=p;t.pose.id=nextId++;t.hasPosition=false;tracks.push_back(t);j=int(tracks.size()-1);}
        used.insert(j);Track& t=tracks[j];if(now-t.seen>.3){t.pending=-1;t.count=0;}t.seen=now;
        if(t.pending!=p.state){t.pending=p.state;t.count=1;t.pendingSince=now;}else t.count++;
        int state=t.pose.state,id=t.pose.id;if(t.count>=3&&now-t.pendingSince>=.12)state=p.state;
        bool oldPosition=t.hasPosition;cv::Vec3d old=t.pose.xyz;double elapsed=oldPosition?now-t.positionTime:1e20;
        t.valid=false;
        if(p.measured&&(!oldPosition||elapsed>=.5||cv::norm(p.xyz-old)<=std::max(120.,2000*elapsed))){
            double alpha=elapsed<.5?1-std::exp(-elapsed/.12):1.;old=oldPosition?old*(1-alpha)+p.xyz*alpha:p.xyz;
            t.hasPosition=true;t.positionTime=now;t.valid=true;}
        t.pose=p;t.pose.xyz=old;t.pose.state=state;t.pose.id=id;p.id=id;
    }
    for(size_t j=0;j<tracks.size();j++)if(!used.count(j)){tracks[j].valid=false;tracks[j].pose.confidence=0;tracks[j].pose.rawState=-1;tracks[j].pending=-1;tracks[j].count=0;}
    // Bound stale ID history on the embedded device without changing active IDs.
    while(tracks.size()>32){auto it=std::min_element(tracks.begin(),tracks.end(),[](const Track&a,const Track&b){return a.seen<b.seen;});tracks.erase(it);}
    return tracks;
}
std::string resultJson(const std::vector<Track>& tracks) {
    std::ostringstream s;s.imbue(std::locale::classic());s<<"{\"hands\":[";bool comma=false;
    for(const Track& t:tracks){if(comma)s<<',';comma=true;
        s<<"{\"hand_id\":"<<t.pose.id<<",\"xyz_mm\":";
        if(t.hasPosition)s<<'['<<std::fixed<<std::setprecision(1)<<t.pose.xyz[0]<<','<<t.pose.xyz[1]<<','<<t.pose.xyz[2]<<']';else s<<"null";
        s<<",\"state\":\""<<label(t.pose.state)<<"\",\"class_id\":"<<t.pose.state;
        s<<",\"raw_class_name\":";if(t.pose.rawState>=0)s<<'"'<<label(t.pose.rawState)<<'"';else s<<"null";
        s<<",\"confidence_class_name\":";if(t.pose.rawState>=0)s<<'"'<<label(t.pose.rawState)<<'"';else s<<"null";
        s<<",\"confidence\":"<<std::fixed<<std::setprecision(4)<<t.pose.confidence<<",\"valid\":"<<(t.valid?"true":"false")<<'}';}
    s<<"]}";return s.str();
}
}
