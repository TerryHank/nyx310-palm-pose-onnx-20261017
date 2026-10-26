#pragma once
#include <array>
#include <vector>
#include <string>
#include <opencv2/core.hpp>

namespace hand {
using Joints=std::array<cv::Vec3d,21>;
using Outputs=std::vector<std::vector<float>>;
struct Palm { cv::Vec4d box; std::array<cv::Point2d,7> points; double score; };
struct Crop { cv::Mat input, rotation; cv::Vec4d box; cv::Point2d bias; double angle=0; };
struct Pose { Joints points,world; cv::Point2d reference; double confidence=0; int side=0,state=0,id=0; bool measured=false; cv::Vec3d xyz; };
cv::Mat palmInput(const cv::Mat& rgb, cv::Point2d& bias);
std::vector<Palm> decodePalms(const Outputs&, cv::Size, cv::Point2d,double,double);
Crop poseInput(const cv::Mat& rgb, const Palm&);
bool decodePose(const Outputs&, const Crop&, double palmScore, Pose&);
bool classify(Pose&,cv::Vec3d* scores=nullptr);
bool measure(Pose&, const cv::Mat& depth, const cv::Matx33d& K, const cv::Vec<double,8>& distortion, bool matched);
Palm trackedPalm(const Pose&);
const char* label(int state);
struct Track {
    Pose pose; double seen=0,positionTime=0,pendingSince=0; int pending=-1,count=0;
    bool hasPosition=false,valid=false;
};
class Tracker {
    std::vector<Track> tracks; int nextId=1;
public:
    std::vector<Track> update(std::vector<Pose>&,double,cv::Size);
};
std::string resultJson(const std::vector<Track>&);
}
