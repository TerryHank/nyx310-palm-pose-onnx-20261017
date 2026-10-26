#include "hagrid.h"
#include <cstring>
#define EXPORT extern "C" __attribute__((visibility("default")))
EXPORT int core_gesture_decode(const float** inputs,int w,int h,double* output){
    try{
        hand::Outputs tensors;for(int branch=0;branch<3;branch++){int area=(52>>branch)*(40>>branch);
            tensors.emplace_back(inputs[branch*2],inputs[branch*2]+64*area);tensors.emplace_back(inputs[branch*2+1],inputs[branch*2+1]+34*area);}
        auto gestures=hand::decodeGestures(tensors,{w,h});int j=0;
        for(auto g:gestures){for(int i=0;i<4;i++)output[j++]=g.box[i];output[j++]=g.score;output[j++]=g.cls;}
        return gestures.size();
    }catch(...){return -1;}
}
EXPORT int core_gesture_input(const uint8_t* rgb,int w,int h,uint8_t* output){
    try{auto input=hand::gestureInput(cv::Mat(h,w,CV_8UC3,const_cast<uint8_t*>(rgb)));std::memcpy(output,input.data,416*320*3);return 0;}catch(...){return -1;}
}
