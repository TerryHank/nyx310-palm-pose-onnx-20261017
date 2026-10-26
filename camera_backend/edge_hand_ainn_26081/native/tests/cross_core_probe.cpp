#include "hand_core.h"
#include <fstream>
#include <iostream>
#include <cstdint>
#include <stdexcept>
#include <cmath>
#include <algorithm>
template<class T>void read(std::istream& s,T* p,size_t n){if(!s.read(reinterpret_cast<char*>(p),sizeof(T)*n))throw std::runtime_error("truncated fixtures");}
int main(int argc,char** argv){
    try{
        if(argc!=2)return 2;std::ifstream f(argv[1],std::ios::binary);uint32_t counts[3];read(f,counts,3);
        if(counts[0]>10000||counts[1]>100||counts[2]>100)return 3;
        double scoreError=0,xyzError=0,pixelMean=0,pixelMax=0;
        for(uint32_t i=0;i<counts[0];i++){
            hand::Pose p;int32_t state;cv::Vec3d expected,actual;
            read(f,p.world[0].val,63);read(f,p.points[0].val,63);read(f,&state,1);read(f,expected.val,3);
            if(!hand::classify(p,&actual)||p.state!=state)throw std::runtime_error("ARM classification mismatch");
            for(int j=0;j<3;j++)scoreError=std::max(scoreError,std::abs(actual[j]-expected[j]));
        }
        if(scoreError>1e-8)throw std::runtime_error("ARM classification scores differ");
        for(uint32_t i=0;i<counts[1];i++){
            int32_t meta[4];read(f,meta,4);if(meta[0]<=0||meta[1]<=0||meta[0]>4096||meta[1]>4096)return 4;
            hand::Pose p;cv::Matx33d K;cv::Vec<double,8>D;cv::Vec3d expected;
            read(f,p.points[0].val,63);read(f,K.val,9);read(f,D.val,8);read(f,expected.val,3);
            cv::Mat depth(meta[1],meta[0],CV_16UC1);read(f,depth.ptr<uint16_t>(),depth.total());
            bool valid=hand::measure(p,depth,K,D,meta[2]);if(valid!=bool(meta[3]))throw std::runtime_error("ARM depth acceptance mismatch");
            if(valid)for(int j=0;j<3;j++)xyzError=std::max(xyzError,std::abs(p.xyz[j]-expected[j]));
        }
        if(xyzError>1e-4)throw std::runtime_error("ARM XYZ differs");
        for(uint32_t i=0;i<counts[2];i++){
            int32_t size[2];read(f,size,2);if(size[0]<=0||size[1]<=0||size[0]>4096||size[1]>4096)return 5;
            cv::Mat rgb(size[1],size[0],CV_8UC3);read(f,rgb.ptr<uint8_t>(),rgb.total()*3);
            double rawPalm[19];read(f,rawPalm,19);cv::Mat expectedPalm(192,192,CV_8UC3),expectedPose(224,224,CV_8UC3);
            read(f,expectedPalm.ptr<uint8_t>(),expectedPalm.total()*3);read(f,expectedPose.ptr<uint8_t>(),expectedPose.total()*3);
            cv::Point2d bias;cv::Mat palmInput=hand::palmInput(rgb,bias);hand::Palm palm;
            for(int j=0;j<4;j++)palm.box[j]=rawPalm[j];for(int j=0;j<7;j++)palm.points[j]={rawPalm[4+j*2],rawPalm[5+j*2]};palm.score=rawPalm[18];
            auto crop=hand::poseInput(rgb,palm);if(crop.input.empty())throw std::runtime_error("ARM pose crop empty");
            for(auto pair:{std::make_pair(palmInput,expectedPalm),std::make_pair(crop.input,expectedPose)}){
                double sum=0;auto* a=pair.first.ptr<uint8_t>();auto* b=pair.second.ptr<uint8_t>();size_t n=pair.first.total()*3;
                for(size_t j=0;j<n;j++){double error=std::abs(int(a[j])-int(b[j]));sum+=error;pixelMax=std::max(pixelMax,error);}
                pixelMean=std::max(pixelMean,sum/n);
            }
        }
        if(pixelMean>.5||pixelMax>3)throw std::runtime_error("ARM preprocessing differs");
        std::cout<<"{\"passed\":true,\"architecture\":\"aarch64\",\"classification_cases\":"<<counts[0]<<",\"depth_cases\":"<<counts[1]<<",\"preprocess_cases\":"<<counts[2]
                 <<",\"max_score_error\":"<<scoreError<<",\"max_xyz_error_mm\":"<<xyzError<<",\"max_pixel_error\":"<<pixelMax<<",\"max_pixel_mean_error\":"<<pixelMean<<",\"npu_initialized\":false}"<<std::endl;
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
