#include "alg_api.h"
#include "hand_core.h"
#include "rknn_api.h"
#include <json.h>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <stdexcept>
#include <opencv2/imgproc.hpp>

namespace {
void check(int ret,const char* operation){if(ret)throw std::runtime_error(std::string(operation)+":"+std::to_string(ret));}
class Model {
    rknn_context context=0;int size=0;std::vector<uint32_t> elements;
public:
    ~Model(){if(context)rknn_destroy(context);}
    void open(const std::string& path,int inputSize,const std::vector<uint32_t>& expected){
        if(!std::ifstream(path).good())throw std::runtime_error("model file missing: "+path);
        check(rknn_init(&context,const_cast<char*>(path.c_str()),0,0,nullptr),"rknn_init");
        rknn_input_output_num count{};check(rknn_query(context,RKNN_QUERY_IN_OUT_NUM,&count,sizeof(count)),"rknn_query IO");
        if(count.n_input!=1||count.n_output!=expected.size())throw std::runtime_error("model IO count mismatch");
        rknn_tensor_attr input{};check(rknn_query(context,RKNN_QUERY_INPUT_ATTR,&input,sizeof(input)),"rknn_query input");
        if(input.n_elems!=uint32_t(inputSize*inputSize*3))throw std::runtime_error("model input shape mismatch");
        for(uint32_t i=0;i<count.n_output;i++){
            rknn_tensor_attr attr{};attr.index=i;check(rknn_query(context,RKNN_QUERY_OUTPUT_ATTR,&attr,sizeof(attr)),"rknn_query output");
            if(attr.n_elems!=expected[i])throw std::runtime_error("model output shape mismatch");elements.push_back(attr.n_elems);
        }
        size=inputSize;
    }
    hand::Outputs infer(const cv::Mat& image){
        if(image.type()!=CV_8UC3||image.rows!=size||image.cols!=size||!image.isContinuous())throw std::runtime_error("model input format mismatch");
        rknn_input input{};input.index=0;input.buf=image.data;input.size=size*size*3;input.type=RKNN_TENSOR_UINT8;input.fmt=RKNN_TENSOR_NHWC;input.pass_through=0;
        check(rknn_inputs_set(context,1,&input),"rknn_inputs_set");check(rknn_run(context,nullptr),"rknn_run");
        std::vector<rknn_output> outputs(elements.size());for(size_t i=0;i<outputs.size();i++){outputs[i].index=i;outputs[i].want_float=1;}
        check(rknn_outputs_get(context,outputs.size(),outputs.data(),nullptr),"rknn_outputs_get");
        hand::Outputs copied;
        try {
            for(size_t i=0;i<outputs.size();i++){
                if(!outputs[i].buf||outputs[i].size<elements[i]*sizeof(float))throw std::runtime_error("model output buffer too small");
                auto* p=static_cast<float*>(outputs[i].buf);copied.emplace_back(p,p+elements[i]);
                for(float v:copied.back())if(!std::isfinite(v))throw std::runtime_error("nonfinite NPU output");
            }
        }catch(...){rknn_outputs_release(context,outputs.size(),outputs.data());throw;}
        check(rknn_outputs_release(context,outputs.size(),outputs.data()),"rknn_outputs_release");return copied;
    }
};
struct Frame {cv::Mat rgb,depth;LENS_PARAMETERS_T lens{};uint64_t timestamp=0;bool matched=false;};
class Algorithm {
    Model palm,pose;std::mutex mutex;std::condition_variable cv;std::thread worker;
    bool running=false,available=false;Frame pending;
    std::function<void(uint64_t,const char*,uint32_t)> callback;
    hand::Tracker tracker;std::vector<hand::Palm> previous;
    cv::Vec<double,8> distortion{};cv::Size calibrationSize{1600,1200};
    cv::Vec4d expectedIntrinsics;bool checkCalibration=false;
    uint64_t lastTimestamp=0;
public:
    std::atomic<double> threshold{.6},nms{.3};
    ~Algorithm(){stop();}
    void initialize(const ALGO_INIT_PARAM_T& parameters){
        std::string root=parameters.p_alg_root_dir;if(root.back()!='/')root+='/';
        callback=parameters.process_cb_func;
        std::ifstream file(root+"config/hand.json");Json::Value config;Json::CharReaderBuilder reader;std::string errors;
        if(!file||!Json::parseFromStream(reader,file,&config,&errors))throw std::runtime_error("invalid config/hand.json");
        auto D=config["ColorDistortion"];if(!D.isArray()||D.size()!=8)throw std::runtime_error("missing distortion coefficients");
        for(int j=0;j<8;j++){distortion[j]=D[j].asDouble();if(!std::isfinite(distortion[j]))throw std::runtime_error("nonfinite distortion");}
        calibrationSize={config["CalibrationWidth"].asInt(),config["CalibrationHeight"].asInt()};
        if(calibrationSize.width<1||calibrationSize.height<1)throw std::runtime_error("invalid calibration image size");
        auto intr=config["ExpectedColorIntrinsics"];if(!intr.isArray()||intr.size()!=4)throw std::runtime_error("missing color intrinsics");
        for(int i=0;i<4;i++)expectedIntrinsics[i]=intr[i].asDouble();checkCalibration=true;
        palm.open(root+"model/palm.rknn",192,{2016*18,2016});
        pose.open(root+"model/handpose.rknn",224,{63,1,1,63});
    }
    int start(){
        std::lock_guard<std::mutex> lock(mutex);if(running)return 0;
        tracker=hand::Tracker();previous.clear();lastTimestamp=0;available=false;running=true;
        worker=std::thread([this]{loop();});return 0;
    }
    int stop(){
        {std::lock_guard<std::mutex> lock(mutex);if(worker.joinable()&&worker.get_id()==std::this_thread::get_id())return ALGO_RET_NOT_SUPPORT;running=false;available=false;}
        cv.notify_all();if(worker.joinable())worker.join();return 0;
    }
    int push(const CAM_JOINT_FRAME_T* joint){
        if(!joint||!joint->p_frame)return ALGO_RET_CHECK_PARAMS_NG;
        if(joint->frame_cnt<1||joint->frame_cnt>5)return ALGO_RET_CHECK_FRAME_COUNT_NG;
        const CAM_SINGLE_FRAME_T *rgb=nullptr,*depth=nullptr;
        for(int i=0;i<joint->frame_cnt;i++){
            const auto& f=joint->p_frame[i];if(f.frame_type==FRAME_RGB){if(rgb)return ALGO_RET_CHECK_FRAME_INDEX_NG;rgb=&f;}
            if(f.frame_type==FRAME_TRANSFORM_DEPTH_IMG_TO_COLOR_SENSOR){if(depth)return ALGO_RET_CHECK_FRAME_INDEX_NG;depth=&f;}
        }
        auto valid=[](const CAM_SINGLE_FRAME_T* f,int bytes,PIXEL_FORMAT_E fmt){return f&&f->p_data&&f->width>=2&&f->height>=2&&f->width<=4096&&f->height<=4096&&f->pixel_format==fmt&&uint64_t(f->data_len)==uint64_t(f->width)*f->height*bytes;};
        if(!valid(rgb,3,PIXEL_FORMAT_RGB_888))return ALGO_RET_CHECK_FRAME_RGB_PARAMS_NG;
        if(depth&&(!valid(depth,2,PIXEL_FORMAT_DEPTH_MM16)||depth->width!=rgb->width||depth->height!=rgb->height))return ALGO_RET_CHECK_FRAME_DEPTH_PARAMS_NG;
        std::lock_guard<std::mutex> lock(mutex);if(!running)return ALGO_RET_NG;
        if(rgb->timestamp<=lastTimestamp&&lastTimestamp!=0)return ALGO_RET_CHECK_FRAME_NG;
        pending.rgb=cv::Mat(rgb->height,rgb->width,CV_8UC3,rgb->p_data).clone();
        pending.depth=depth?cv::Mat(depth->height,depth->width,CV_16UC1,depth->p_data).clone():cv::Mat();
        pending.lens=joint->lens_parameters;pending.timestamp=rgb->timestamp;
        pending.matched=depth&&depth->timestamp==rgb->timestamp;
        available=true;lastTimestamp=rgb->timestamp;cv.notify_one();return 0;
    }
    void emit(uint64_t timestamp,const std::string& text){
        if(callback)callback(timestamp,text.c_str(),uint32_t(text.size()+1));
    }
    void process(Frame& f){
        int width=std::min(f.rgb.cols,640),height=cvRound(double(f.rgb.rows)*width/f.rgb.cols);
        cv::Mat small;cv::resize(f.rgb,small,{width,height});cv::Point2d bias;
        cv::Mat input=hand::palmInput(small,bias);
        auto palms=hand::decodePalms(palm.infer(input),small.size(),bias,threshold,nms);
        if(palms.empty())palms=previous;previous.clear();std::vector<hand::Pose> hands;
        cv::Vec4d intr(f.lens.colorIntrinsic[0],f.lens.colorIntrinsic[1],f.lens.colorIntrinsic[2],f.lens.colorIntrinsic[3]);
        if(intr[0]>=f.rgb.cols||intr[1]>=f.rgb.rows){intr[0]*=double(f.rgb.cols)/calibrationSize.width;intr[2]*=double(f.rgb.cols)/calibrationSize.width;
            intr[1]*=double(f.rgb.rows)/calibrationSize.height;intr[3]*=double(f.rgb.rows)/calibrationSize.height;}
        bool calibrated=intr[2]>0&&intr[3]>0;
        for(int i=0;i<4;i++){
            double scaled=expectedIntrinsics[i]*(i%2==0?double(f.rgb.cols)/calibrationSize.width:double(f.rgb.rows)/calibrationSize.height);
            if(!std::isfinite(intr[i])||(checkCalibration&&std::abs(intr[i]-scaled)>std::max(3.,std::abs(scaled)*.02)))calibrated=false;
        }
        cv::Matx33d K(intr[2],0,intr[0],0,intr[3],intr[1],0,0,1);
        for(const auto& p:palms){
            if(hands.size()==2)break;auto crop=hand::poseInput(small,p);if(crop.input.empty())continue;
            hand::Pose h;if(!hand::decodePose(pose.infer(crop.input),crop,p.score,h))continue;
            previous.push_back(hand::trackedPalm(h));
            for(auto& v:h.points){v[0]*=double(f.rgb.cols)/width;v[1]*=double(f.rgb.rows)/height;v[2]*=double(f.rgb.cols)/width;}
            if(!hand::classify(h))continue;
            hand::measure(h,f.depth,K,distortion,f.matched&&calibrated);hands.push_back(h);
        }
        emit(f.timestamp,hand::resultJson(tracker.update(hands,f.timestamp/1000.,f.rgb.size())));
    }
    void loop(){
        for(;;){Frame f;
            {std::unique_lock<std::mutex> lock(mutex);cv.wait(lock,[this]{return !running||available;});if(!running)break;f=std::move(pending);available=false;}
            try{process(f);}catch(const std::exception& e){
                std::cerr<<"Hand pipeline: "<<e.what()<<std::endl;previous.clear();
                std::vector<hand::Pose> none;auto json=hand::resultJson(tracker.update(none,f.timestamp/1000.,f.rgb.size()));
                json.insert(1,"\"error\":\"inference_failed\",");try{emit(f.timestamp,json);}catch(...){std::cerr<<"Hand callback failed"<<std::endl;}
            }catch(...){std::cerr<<"Hand pipeline unexpected error"<<std::endl;}
        }
    }
};
std::mutex apiMutex;std::unique_ptr<Algorithm> algorithm;std::string returned;
}

ALGO_RET_E algo_initialize(const ALGO_INIT_PARAM_T* p){
    std::lock_guard<std::mutex> lock(apiMutex);
    if(!p||!p->p_alg_root_dir||!p->p_alg_root_dir[0]||!p->process_cb_func)return ALGO_RET_CHECK_PARAMS_NG;
    if(algorithm)return ALGO_RET_NG;
    try{std::unique_ptr<Algorithm> a(new Algorithm);a->initialize(*p);algorithm=std::move(a);return ALGO_RET_OK;}
    catch(const std::exception& e){std::cerr<<"Hand initialize: "<<e.what()<<std::endl;return ALGO_RET_INIT_MODEL_NG;}
    catch(...){return ALGO_RET_INIT_MODEL_NG;}
}
ALGO_RET_E algo_ctrl_start(){std::lock_guard<std::mutex> lock(apiMutex);try{return algorithm?ALGO_RET_E(algorithm->start()):ALGO_RET_NOT_INIT;}catch(...){return ALGO_RET_NG;}}
ALGO_RET_E algo_ctrl_stop(){std::lock_guard<std::mutex> lock(apiMutex);return algorithm?ALGO_RET_E(algorithm->stop()):ALGO_RET_NOT_INIT;}
ALGO_RET_E algo_push_frame(const CAM_JOINT_FRAME_T* p){std::lock_guard<std::mutex> lock(apiMutex);try{return algorithm?ALGO_RET_E(algorithm->push(p)):ALGO_RET_NOT_INIT;}catch(...){return ALGO_RET_CHECK_FRAME_NG;}}
ALGO_RET_E algo_release(){std::lock_guard<std::mutex> lock(apiMutex);if(!algorithm)return ALGO_RET_NOT_INIT;int ret=algorithm->stop();if(ret)return ALGO_RET_E(ret);algorithm.reset();return ALGO_RET_OK;}
ALGO_RET_E algo_set_output_handle(OutputHandleType,void*){return ALGO_RET_NOT_SUPPORT;}
ALGO_RET_E algo_get_param(const char* input,uint16_t length,const char** output,uint16_t* outLength){
    std::lock_guard<std::mutex> lock(apiMutex);if(!algorithm)return ALGO_RET_NOT_INIT;
    if(!input||length<4||!output||!outLength)return ALGO_RET_CHECK_PARAMS_NG;uint32_t id;std::memcpy(&id,input,4);
    if(id==PARAM_VERSION)returned="26.8.1-hand.1";else if(id==PARAM_NAME)returned="PalmPoseGraspXYZ";
    else if(id==PARAM_CONF_THRESH||id==PARAM_IOU_THRESH){std::ostringstream s;s<<"{\""<<(id==PARAM_CONF_THRESH?"Conf_Thresh":"IOU_Thresh")<<"\":"<<(id==PARAM_CONF_THRESH?algorithm->threshold.load():algorithm->nms.load())<<'}';returned=s.str();}
    else return ALGO_RET_NOT_SUPPORT;*output=returned.c_str();*outLength=uint16_t(returned.size()+1);return ALGO_RET_OK;
}
ALGO_RET_E algo_set_param(const char* input,uint16_t length,const char**,uint16_t*){
    std::lock_guard<std::mutex> lock(apiMutex);if(!algorithm)return ALGO_RET_NOT_INIT;
    if(!input||length<4)return ALGO_RET_CHECK_PARAMS_NG;uint32_t id;std::memcpy(&id,input,4);
    if(id==PARAM_SAVE_RESULT_IMAGE_ENABLE&&length==5)return input[4]?ALGO_RET_NOT_SUPPORT:ALGO_RET_OK;
    if(id!=PARAM_CONF_THRESH&&id!=PARAM_IOU_THRESH)return ALGO_RET_NOT_SUPPORT;
    std::string text(input+4,length-4);if(!text.empty()&&text.back()==0)text.pop_back();
    Json::Value json;Json::CharReaderBuilder builder;std::string error;std::istringstream stream(text);
    if(!Json::parseFromStream(builder,stream,&json,&error))return ALGO_RET_JSON_PARSE_NG;
    const char* key=id==PARAM_CONF_THRESH?"Conf_Thresh":"IOU_Thresh";
    if(!json[key].isNumeric())return ALGO_RET_CHECK_PARAMS_NG;double value=json[key].asDouble();
    if(!std::isfinite(value)||value<=0||value>=1)return ALGO_RET_CHECK_PARAMS_NG;
    if(id==PARAM_CONF_THRESH)algorithm->threshold=value;else algorithm->nms=value;return ALGO_RET_OK;
}
