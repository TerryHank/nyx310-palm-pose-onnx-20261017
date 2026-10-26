#include "hagrid.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <iomanip>

namespace hand {
const char* gestureName(int c){return c==0?"GRASP":"OPEN";}
static void dimensions(cv::Size size,double& ratio,int& w,int& h,int& left,int& top){
    ratio=std::min(416./size.width,320./size.height);w=int(std::nearbyint(size.width*ratio));h=int(std::nearbyint(size.height*ratio));
    left=int(std::nearbyint((416-w)*.5-.1));top=int(std::nearbyint((320-h)*.5-.1));
}
cv::Mat gestureInput(const cv::Mat& rgb){
    double r;int w,h,left,top;dimensions(rgb.size(),r,w,h,left,top);
    cv::Mat resized,result;cv::resize(rgb,resized,{w,h});
    cv::copyMakeBorder(resized,result,top,320-h-top,left,416-w-left,cv::BORDER_CONSTANT,cv::Scalar(114,114,114));
    return result;
}
static double iou(cv::Vec4d a,cv::Vec4d b){
    double inter=std::max(0.,std::min(a[2],b[2])-std::max(a[0],b[0]))*
                 std::max(0.,std::min(a[3],b[3])-std::max(a[1],b[1]));
    double total=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter;
    return total>0?inter/total:0;
}
std::vector<Gesture> decodeGestures(const Outputs& o,cv::Size size,double threshold){
    if(o.size()!=6)throw std::runtime_error("binary gesture output count");
    static const int group[34]={0,0,1,0,0,1,1,0,0,0,0,0,-1,0,0,1,0,0,0,0,1,0,0,0,1,1,1,1,0,0,1,0,0,-1};
    std::vector<Gesture> candidates;
    for(int branch=0;branch<3;branch++){
        int width=52>>branch,height=40>>branch,area=width*height,stride=8<<branch;
        if(o[branch*2].size()!=size_t(64*area)||o[branch*2+1].size()!=size_t(34*area))throw std::runtime_error("binary gesture output shape");
        const auto& logits=o[branch*2];const auto& scores=o[branch*2+1];
        for(int p=0;p<area;p++){
            Gesture g;
            for(int c=0;c<34;c++){
                double v=scores[c*area+p];g.score=std::max(g.score,v);
                if(group[c]==0)g.graspScore=std::max(g.graspScore,v);
                if(group[c]==1)g.openScore=std::max(g.openScore,v);
            }
            if(!std::isfinite(g.score)||g.score<threshold)continue;
            g.cls=g.graspScore>g.openScore?0:1;
            double d[4];
            for(int side=0;side<4;side++){
                double maximum=-1e30;for(int k=0;k<16;k++)maximum=std::max(maximum,double(logits[(side*16+k)*area+p]));
                double sum=0,weighted=0;for(int k=0;k<16;k++){double v=std::exp(logits[(side*16+k)*area+p]-maximum);sum+=v;weighted+=k*v;}d[side]=weighted/sum;
            }
            double x=p%width+.5,y=p/width+.5;
            g.box={(x-d[0])*stride,(y-d[1])*stride,(x+d[2])*stride,(y+d[3])*stride};
            if(g.box[2]>g.box[0]&&g.box[3]>g.box[1])candidates.push_back(g);
        }
    }
    std::stable_sort(candidates.begin(),candidates.end(),[](const Gesture&a,const Gesture&b){return a.score>b.score;});
    std::vector<Gesture> suppressed;
    for(auto g:candidates){
        bool duplicate=false;for(auto other:suppressed)if(iou(other.box,g.box)>.35){duplicate=true;break;}
        if(!duplicate)suppressed.push_back(g);
        if(suppressed.size()==2)break;
    }
    if(suppressed.empty())return {};
    Gesture g=suppressed.front();double ratio;int w,h,left,top;dimensions(size,ratio,w,h,left,top);
    for(int i=0;i<4;i++)g.box[i]=std::max(0.,std::min(double(i%2?size.height:size.width),(g.box[i]-(i%2?top:left))/ratio));
    return {g};
}
void PresenceState::clear(){hasPosition=false;hasState=false;pending=-1;count=0;}
std::string PresenceState::result(const Gesture* gesture,const Pose* pose,double now,const std::string& timings){
    std::ostringstream s;s.imbue(std::locale::classic());s<<"{\"hand_present\":"<<(gesture?"true":"false")<<",\"hands\":[";
    if(!gesture){clear();s<<']';if(!timings.empty())s<<",\"timing_ms\":"<<timings;s<<'}';return s.str();}
    int candidate=gesture->cls;
    if(!hasState){state=candidate;hasState=true;}
    if(candidate!=pending){pending=candidate;count=1;pendingSince=now;}else count++;
    if(count>=3&&now-pendingSince>=.12)state=candidate;
    bool valid=false;
    if(pose&&pose->measured){
        double elapsed=hasPosition?now-positionTime:1e20;
        if(!hasPosition||elapsed>=.5||cv::norm(pose->xyz-xyz)<=std::max(120.,2000*elapsed)){
            double alpha=elapsed<.5?1-std::exp(-elapsed/.12):1.;xyz=hasPosition?xyz*(1-alpha)+pose->xyz*alpha:pose->xyz;
            hasPosition=true;positionTime=now;valid=true;
        }
    }
    s<<"{\"xyz_mm\":";
    if(hasPosition)s<<'['<<std::fixed<<std::setprecision(1)<<xyz[0]<<','<<xyz[1]<<','<<xyz[2]<<']';else s<<"null";
    s<<",\"state\":\""<<gestureName(state)<<"\",\"class_id\":"<<state;
    s<<",\"confidence\":"<<std::fixed<<std::setprecision(4)<<gesture->score<<",\"valid\":"<<(valid?"true":"false");
    s<<",\"grasp_score\":"<<gesture->graspScore<<",\"open_score\":"<<gesture->openScore;
    s<<",\"position_status\":\""<<(valid?"VALID":hasPosition?"HOLD":"NO_DEPTH")<<"\"}]";
    if(!timings.empty())s<<",\"timing_ms\":"<<timings;
    s<<'}';return s.str();
}
}
