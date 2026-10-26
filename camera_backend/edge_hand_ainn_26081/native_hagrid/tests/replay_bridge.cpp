// Host test seam for the unchanged manufacturer's C++ std::function callback ABI.
#include "alg_api.h"
#include "hand_core.h"
#include <cstring>
#define EXPORT extern "C" __attribute__((visibility("default")))
using Result=void(*)(uint64_t,const char*,uint32_t);
EXPORT int replay_initialize(const char* root,Result result){
    ALGO_INIT_PARAM_T p{};p.p_alg_root_dir=const_cast<char*>(root);
    p.process_cb_func=[result](uint64_t t,const char* text,uint32_t size){result(t,text,size);};return algo_initialize(&p);
}
EXPORT int replay_push(uint8_t* rgb,uint16_t* depth,int w,int h,uint64_t time,uint64_t depthTime,const float* lens){
    CAM_SINGLE_FRAME_T frames[2]{};
    frames[0]={1,time,FRAME_RGB,PIXEL_FORMAT_RGB_888,uint32_t(w),uint32_t(h),uint32_t(w*h*3),rgb};
    frames[1]={1,depthTime,FRAME_TRANSFORM_DEPTH_IMG_TO_COLOR_SENSOR,PIXEL_FORMAT_DEPTH_MM16,uint32_t(w),uint32_t(h),uint32_t(w*h*2),reinterpret_cast<uint8_t*>(depth)};
    CAM_JOINT_FRAME_T joint{};joint.frame_cnt=depth?2:1;joint.p_frame=frames;std::memcpy(&joint.lens_parameters,lens,sizeof(LENS_PARAMETERS_T));
    return algo_push_frame(&joint);
}
EXPORT int replay_push_raw(uint8_t* rgb,int w,int h,uint16_t* depth,int dw,int dh,uint64_t time,uint64_t depthTime,const float* lens){
    CAM_SINGLE_FRAME_T frames[2]{};
    frames[0]={1,depthTime,FRAME_DEPTH,PIXEL_FORMAT_DEPTH_MM16,uint32_t(dw),uint32_t(dh),uint32_t(dw*dh*2),reinterpret_cast<uint8_t*>(depth)};
    frames[1]={1,time,FRAME_RGB,PIXEL_FORMAT_RGB_888,uint32_t(w),uint32_t(h),uint32_t(w*h*3),rgb};
    CAM_JOINT_FRAME_T joint{};joint.frame_cnt=2;joint.p_frame=frames;std::memcpy(&joint.lens_parameters,lens,sizeof(LENS_PARAMETERS_T));
    return algo_push_frame(&joint);
}
EXPORT int core_measure_raw(const double* points,const uint16_t* depth,const float* rays,int dw,int dh,int w,int h,
                            const double* k,const double* d,const double* r,const double* t,int matched,double* out){
    hand::Pose p;std::memcpy(p.points.data(),points,63*sizeof(double));cv::Matx33d K,R;cv::Vec<double,8>D;cv::Vec3d T;
    std::memcpy(K.val,k,9*sizeof(double));std::memcpy(D.val,d,8*sizeof(double));std::memcpy(R.val,r,9*sizeof(double));std::memcpy(T.val,t,3*sizeof(double));
    bool ok=hand::measureRaw(p,cv::Mat(dh,dw,CV_16UC1,const_cast<uint16_t*>(depth)),cv::Mat(dh,dw,CV_32FC2,const_cast<float*>(rays)),
                             {w,h},K,D,R,T,matched);
    if(ok)std::memcpy(out,p.xyz.val,3*sizeof(double));return ok;
}
EXPORT int core_classify(const double* world,const double* points,double* out){
    hand::Pose p;std::memcpy(p.world.data(),world,63*sizeof(double));std::memcpy(p.points.data(),points,63*sizeof(double));cv::Vec3d scores;
    if(!hand::classify(p,&scores))return -1;std::memcpy(out,scores.val,3*sizeof(double));return p.state;
}
EXPORT int core_measure(const double* points,const uint16_t* depth,int w,int h,const double* k,const double* d,int matched,double* out){
    hand::Pose p;std::memcpy(p.points.data(),points,63*sizeof(double));cv::Matx33d K;cv::Vec<double,8>D;
    std::memcpy(K.val,k,9*sizeof(double));std::memcpy(D.val,d,8*sizeof(double));
    bool ok=hand::measure(p,cv::Mat(h,w,CV_16UC1,const_cast<uint16_t*>(depth)),K,D,matched);
    if(ok)std::memcpy(out,p.xyz.val,3*sizeof(double));return ok;
}
EXPORT int core_palm_input(const uint8_t* rgb,int w,int h,uint8_t* out,double* bias){
    try{cv::Point2d b;cv::Mat input=hand::palmInput(cv::Mat(h,w,CV_8UC3,const_cast<uint8_t*>(rgb)),b);std::memcpy(out,input.data,192*192*3);bias[0]=b.x;bias[1]=b.y;return 0;}catch(...){return -1;}
}
static hand::Palm palmFrom(const double* p){hand::Palm palm;for(int i=0;i<4;i++)palm.box[i]=p[i];for(int i=0;i<7;i++)palm.points[i]={p[4+i*2],p[5+i*2]};palm.score=p[18];return palm;}
EXPORT int core_palm_decode(const float* boxes,const float* logits,int w,int h,const double* bias,double* out){
    try{hand::Outputs outputs{std::vector<float>(boxes,boxes+2016*18),std::vector<float>(logits,logits+2016)};
        auto palms=hand::decodePalms(outputs,{w,h},{bias[0],bias[1]},.6,.3);int j=0;
        for(const auto& p:palms){for(int i=0;i<4;i++)out[j++]=p.box[i];for(auto v:p.points){out[j++]=v.x;out[j++]=v.y;}out[j++]=p.score;}return palms.size();}catch(...){return -1;}
}
EXPORT int core_pose_input(const uint8_t* rgb,int w,int h,const double* palm,uint8_t* out,double* meta){
    try{auto crop=hand::poseInput(cv::Mat(h,w,CV_8UC3,const_cast<uint8_t*>(rgb)),palmFrom(palm));if(crop.input.empty())return -1;
        std::memcpy(out,crop.input.data,224*224*3);for(int j=0;j<4;j++)meta[j]=crop.box[j];meta[4]=crop.angle;
        std::memcpy(meta+5,crop.rotation.ptr<double>(),6*sizeof(double));meta[11]=crop.bias.x;meta[12]=crop.bias.y;return 0;}catch(...){return -1;}
}
EXPORT int core_pose_decode(const float* landmarks,float presence,float side,const float* world,const double* meta,double palmScore,double* out){
    hand::Crop crop;for(int j=0;j<4;j++)crop.box[j]=meta[j];crop.angle=meta[4];crop.rotation=cv::Mat(2,3,CV_64F,const_cast<double*>(meta+5)).clone();crop.bias={meta[11],meta[12]};
    hand::Outputs o{std::vector<float>(landmarks,landmarks+63),{presence},{side},std::vector<float>(world,world+63)};hand::Pose p;
    if(!hand::decodePose(o,crop,palmScore,p))return -1;std::memcpy(out,p.points.data(),63*sizeof(double));std::memcpy(out+63,p.world.data(),63*sizeof(double));out[126]=p.confidence;out[127]=p.side;return 0;
}
