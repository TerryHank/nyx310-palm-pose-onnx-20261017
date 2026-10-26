#pragma once
#include "hand_core.h"
namespace hand {
struct Gesture {cv::Vec4d box;int cls=0;double score=0;};
cv::Mat gestureInput(const cv::Mat& rgb);
std::vector<Gesture> decodeGestures(const Outputs&,cv::Size,double threshold=.25);
const char* gestureName(int);
}
