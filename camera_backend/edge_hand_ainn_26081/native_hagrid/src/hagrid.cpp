#include "hagrid.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hand {
const char* gestureName(int c){
    static const char* names[]={"grabbing","grip","holy","point","call","three3","timeout","xsign",
        "hand_heart","hand_heart2","little_finger","middle_finger","take_picture","dislike","fist","four",
        "like","mute","ok","one","palm","peace","peace_inverted","rock","stop","stop_inverted",
        "three","three2","two_up","two_up_inverted","three_gun","thumb_index","thumb_index2","no_gesture"};
    return c>=0&&c<34?names[c]:"no_gesture";
}
static void dimensions(cv::Size size,double& ratio,int& w,int& h,int& left,int& top){
    ratio=std::min(640./size.width,480./size.height);w=int(std::nearbyint(size.width*ratio));h=int(std::nearbyint(size.height*ratio));
    left=int(std::nearbyint((640-w)*.5-.1));top=int(std::nearbyint((480-h)*.5-.1));
}
cv::Mat gestureInput(const cv::Mat& rgb){
    double r;int w,h,left,top;dimensions(rgb.size(),r,w,h,left,top);
    cv::Mat resized,result;cv::resize(rgb,resized,{w,h});
    cv::copyMakeBorder(resized,result,top,480-h-top,left,640-w-left,cv::BORDER_CONSTANT,cv::Scalar(114,114,114));
    return result;
}
static double iou(cv::Vec4d a,cv::Vec4d b){
    double inter=std::max(0.,std::min(a[2],b[2])-std::max(a[0],b[0]))*
                 std::max(0.,std::min(a[3],b[3])-std::max(a[1],b[1]));
    double total=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter;
    return total>0?inter/total:0;
}
std::vector<Gesture> decodeGestures(const Outputs& o,cv::Size size,double threshold){
    if(o.size()!=6)throw std::runtime_error("HaGRID output count");
    std::vector<Gesture> candidates;
    for(int branch=0;branch<3;branch++){
        int grid=80>>branch,area=grid*(60>>branch),stride=8<<branch;
        if(o[branch*2].size()!=size_t(64*area)||o[branch*2+1].size()!=size_t(34*area))throw std::runtime_error("HaGRID output shape");
        const auto& logits=o[branch*2];const auto& scores=o[branch*2+1];
        for(int p=0;p<area;p++){
            Gesture g;for(int c=0;c<34;c++)if(scores[c*area+p]>g.score){g.score=scores[c*area+p];g.cls=c;}
            if(!std::isfinite(g.score)||g.score<threshold)continue;
            double d[4];
            for(int side=0;side<4;side++){
                double maximum=-1e30;for(int k=0;k<16;k++)maximum=std::max(maximum,double(logits[(side*16+k)*area+p]));
                double sum=0,weighted=0;for(int k=0;k<16;k++){double v=std::exp(logits[(side*16+k)*area+p]-maximum);sum+=v;weighted+=k*v;}d[side]=weighted/sum;
            }
            double x=p%grid+.5,y=p/grid+.5;
            g.box={(x-d[0])*stride,(y-d[1])*stride,(x+d[2])*stride,(y+d[3])*stride};
            if(g.box[2]>g.box[0]&&g.box[3]>g.box[1])candidates.push_back(g);
        }
    }
    std::stable_sort(candidates.begin(),candidates.end(),[](const Gesture&a,const Gesture&b){return a.score>b.score;});
    std::vector<Gesture> selected,result;
    double ratio;int w,h,left,top;dimensions(size,ratio,w,h,left,top);
    for(auto g:candidates){
        bool suppressed=false;for(auto old:selected)if(old.cls==g.cls&&iou(old.box,g.box)>.7){suppressed=true;break;}
        if(suppressed)continue;selected.push_back(g);
        for(int i=0;i<4;i++)g.box[i]=std::max(0.,std::min(double(i%2?size.height:size.width),(g.box[i]-(i%2?top:left))/ratio));
        bool duplicate=false;for(auto old:result)if(iou(old.box,g.box)>.8){duplicate=true;break;}
        if(!duplicate)result.push_back(g);if(selected.size()==2)break;
    }
    return result;
}
}
