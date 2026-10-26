#include "hand_core.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <cmath>
#include <set>
#include <limits>

namespace hand {
static double median(std::vector<double> values){
    size_t n=values.size();std::nth_element(values.begin(),values.begin()+n/2,values.end());double v=values[n/2];
    return n%2?v:(v+*std::max_element(values.begin(),values.begin()+n/2))*.5;
}
bool measureRaw(Pose& p,const cv::Mat& depth,const cv::Mat& rays,cv::Size size,
                const cv::Matx33d& K,const cv::Vec<double,8>& D,
                const cv::Matx33d& R,const cv::Vec3d& t,bool matched){
    p.measured=false;
    p.reference={double((float(p.points[0][0])+float(p.points[9][0]))*.5f),double((float(p.points[0][1])+float(p.points[9][1]))*.5f)};
    if(!matched||depth.empty()||depth.type()!=CV_16UC1||rays.type()!=CV_32FC2||rays.size()!=depth.size())return false;
    if(size.width<2||size.height<2||p.reference.x<0||p.reference.y<0||p.reference.x>=size.width||p.reference.y>=size.height)return false;
    if(!std::isfinite(K(0,0))||!std::isfinite(K(1,1))||K(0,0)<=0||K(1,1)<=0)return false;
    std::vector<cv::Point2f> points,hull;
    for(int i:{0,1,5,9,13,17})points.emplace_back(float(p.points[i][0]),float(p.points[i][1]));
    cv::convexHull(points,hull);if(hull.size()<3)return false;cv::Point2f center;
    for(auto v:hull)center+=v;center*=1.f/hull.size();
    std::vector<cv::Point> polygon;
    for(auto v:hull){v=center+(v-center)*.7f;polygon.emplace_back(cvRound(v.x),cvRound(v.y));}
    cv::Rect bounds=cv::boundingRect(polygon);bounds.x--;bounds.y--;bounds.width+=2;bounds.height+=2;
    bounds&=cv::Rect(0,0,size.width,size.height);if(bounds.empty())return false;
    for(auto& v:polygon)v-=bounds.tl();
    cv::Mat mask=cv::Mat::zeros(bounds.size(),CV_8U);
    cv::fillConvexPoly(mask,polygon,cv::Scalar(1));cv::erode(mask,mask,cv::Mat::ones(3,3,CV_8U));
    int area=cv::countNonZero(mask);if(area<25)return false;
    // Only this contracted palm rectangle has a z-buffer, never a full RGB depth map.
    cv::Mat zbuffer(bounds.size(),CV_32F,cv::Scalar(std::numeric_limits<float>::infinity()));
    for(int y=0;y<depth.rows;y++){
        const auto* mm=depth.ptr<uint16_t>(y);const auto* ray=rays.ptr<cv::Vec2f>(y);
        for(int x=0;x<depth.cols;x++){
            if(mm[x]==0||mm[x]==65535)continue;
            cv::Vec3d point=R*cv::Vec3d(double(ray[x][0])*mm[x],double(ray[x][1])*mm[x],mm[x])+t;
            if(!(point[2]>50&&point[2]<5000))continue;
            double nx=point[0]/point[2],ny=point[1]/point[2],r2=nx*nx+ny*ny,r4=r2*r2,r6=r4*r2;
            double denominator=1+D[5]*r2+D[6]*r4+D[7]*r6;if(std::abs(denominator)<1e-12)continue;
            double radial=(1+D[0]*r2+D[1]*r4+D[4]*r6)/denominator;
            double u=K(0,0)*(nx*radial+2*D[2]*nx*ny+D[3]*(r2+2*nx*nx))+K(0,2);
            double v=K(1,1)*(ny*radial+D[2]*(r2+2*ny*ny)+2*D[3]*nx*ny)+K(1,2);
            if(!std::isfinite(u)||!std::isfinite(v)||u<bounds.x-.5||v<bounds.y-.5||u>=bounds.br().x+.5||v>=bounds.br().y+.5)continue;
            int px=cvRound(u)-bounds.x,py=cvRound(v)-bounds.y;
            if(px<0||py<0||px>=bounds.width||py>=bounds.height||!mask.at<uint8_t>(py,px))continue;
            float& z=zbuffer.at<float>(py,px);z=std::min(z,float(point[2]));
        }
    }
    std::vector<double> values;std::vector<cv::Point> pixels;
    for(int y=0;y<mask.rows;y++)for(int x=0;x<mask.cols;x++)
        if(mask.at<uint8_t>(y,x)&&std::isfinite(zbuffer.at<float>(y,x))){values.push_back(zbuffer.at<float>(y,x));pixels.emplace_back(x+bounds.x,y+bounds.y);}
    if(values.size()<12)return false;double middle=median(values);std::vector<double> deviations;
    for(double v:values)deviations.push_back(std::abs(v-middle));double cutoff=std::max(20.,std::min(80.,3*1.4826*median(deviations)));
    std::vector<double> kept;std::set<int> quadrants;
    for(size_t i=0;i<values.size();i++)if(std::abs(values[i]-middle)<=cutoff){kept.push_back(values[i]);quadrants.insert((pixels[i].x>=center.x)+2*(pixels[i].y>=center.y));}
    if(kept.size()<12||kept.size()<.6*values.size()||kept.size()<.01*area||quadrants.size()<3)return false;
    std::vector<cv::Point2d> ray;
    cv::undistortPoints(std::vector<cv::Point2d>{p.reference},ray,cv::Mat(K),cv::Mat(D));
    double z=median(kept);p.xyz={ray[0].x*z,ray[0].y*z,z};p.measured=true;return true;
}
}
