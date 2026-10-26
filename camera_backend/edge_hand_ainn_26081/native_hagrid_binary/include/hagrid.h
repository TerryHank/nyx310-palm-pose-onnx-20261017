#pragma once
#include "hand_core.h"
namespace hand {
struct Gesture {cv::Vec4d box;int cls=0;double score=0;double graspScore=0,openScore=0;};
cv::Mat gestureInput(const cv::Mat& rgb);
std::vector<Gesture> decodeGestures(const Outputs&,cv::Size,double threshold=.175);
const char* gestureName(int);
class PresenceState {
    bool hasPosition=false,hasState=false;
    cv::Vec3d xyz{0,0,0};double positionTime=0,pendingSince=0;
    int state=1,pending=-1,count=0;
public:
    void clear();
    std::string result(const Gesture*,const Pose*,double now,const std::string& timings="");
};
}
