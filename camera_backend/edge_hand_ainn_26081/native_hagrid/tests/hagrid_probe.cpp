#include "hagrid.h"
#include <fstream>
#include <iostream>
int main(int argc,char**argv){
    if(argc!=2)return 2;std::ifstream file(argv[1],std::ios::binary);uint32_t cases=0;file.read(reinterpret_cast<char*>(&cases),4);
    int checked=0;
    for(uint32_t sample=0;sample<cases;sample++){
        int w,h,n;file.read(reinterpret_cast<char*>(&w),4);file.read(reinterpret_cast<char*>(&h),4);file.read(reinterpret_cast<char*>(&n),4);
        std::vector<double> expected(n*6);file.read(reinterpret_cast<char*>(expected.data()),n*6*8);
        hand::Outputs tensors;for(int i=0;i<6;i++){int area=(80>>(i/2))*(60>>(i/2));tensors.emplace_back((i%2?34:64)*area);file.read(reinterpret_cast<char*>(tensors.back().data()),tensors.back().size()*4);}
        if(!file)return 3;auto result=hand::decodeGestures(tensors,{w,h});if(int(result.size())!=n)return 4;
        for(int i=0;i<n;i++){for(int j=0;j<4;j++)if(std::abs(result[i].box[j]-expected[i*6+j])>1e-4)return 5;
            if(std::abs(result[i].score-expected[i*6+4])>1e-5||result[i].cls!=int(expected[i*6+5]))return 6;}
        checked++;
    }
    std::cout<<"{\"passed\":true,\"cases\":"<<checked<<"}"<<std::endl;
}
