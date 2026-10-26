#include "hand_core.h"
#include <fstream>
#include <iostream>
#include <cstdint>
#include <stdexcept>
#include <cmath>
template<class T>void read(std::istream& stream,T* p,size_t n){if(!stream.read(reinterpret_cast<char*>(p),sizeof(T)*n))throw std::runtime_error("truncated fixture");}
int main(int argc,char** argv){
    try{
        if(argc!=2)return 2;std::ifstream file(argv[1],std::ios::binary);uint32_t count;read(file,&count,1);if(count>100)return 3;
        cv::Mat rays(480,640,CV_32FC2);read(file,rays.ptr<float>(),640*480*2);double maxError=0;int validCases=0;
        for(uint32_t i=0;i<count;i++){
            int32_t flags[2];read(file,flags,2);hand::Pose p;cv::Matx33d K,R;cv::Vec<double,8>D;cv::Vec3d t,expected;
            read(file,p.points[0].val,63);read(file,K.val,9);read(file,D.val,8);read(file,R.val,9);read(file,t.val,3);read(file,expected.val,3);
            cv::Mat depth(480,640,CV_16UC1);read(file,depth.ptr<uint16_t>(),640*480);
            bool valid=hand::measureRaw(p,depth,rays,{640,480},K,D,R,t,flags[0]);
            if(valid!=bool(flags[1]))throw std::runtime_error("ARM raw acceptance differs");
            if(valid){validCases++;for(int j=0;j<3;j++)maxError=std::max(maxError,std::abs(p.xyz[j]-expected[j]));}
        }
        if(maxError>=.002)throw std::runtime_error("ARM raw XYZ differs");
        std::cout<<"{\"passed\":true,\"architecture\":\"aarch64\",\"raw_cases\":"<<count<<",\"valid_cases\":"<<validCases<<",\"max_xyz_error_mm\":"<<maxError<<",\"npu_initialized\":false}"<<std::endl;
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 1;}
}
