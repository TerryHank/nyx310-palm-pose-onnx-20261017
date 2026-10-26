#include "alg_api.h"
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
static_assert(sizeof(CAM_SINGLE_FRAME_T)==40,"SDK frame packing changed");
static_assert(sizeof(LENS_PARAMETERS_T)==80,"SDK calibration packing changed");
static_assert(sizeof(CAM_JOINT_FRAME_T)==89,"SDK joint frame packing changed");
static_assert(sizeof(ALGO_INIT_PARAM_T)==40,"SDK std::function ABI changed");
int main(int argc,char** argv){
    if(argc!=2)return 2;
    void* lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!lib){std::cerr<<dlerror()<<'\n';return 3;}
    for(const char* name:{"algo_initialize","algo_ctrl_start","algo_ctrl_stop","algo_push_frame","algo_set_param","algo_get_param","algo_set_output_handle","algo_release"})
        if(!dlsym(lib,name)){std::cerr<<name<<" missing\n";return 4;}
    auto initialize=reinterpret_cast<decltype(&algo_initialize)>(dlsym(lib,"algo_initialize"));
    auto start=reinterpret_cast<decltype(&algo_ctrl_start)>(dlsym(lib,"algo_ctrl_start"));
    auto release=reinterpret_cast<decltype(&algo_release)>(dlsym(lib,"algo_release"));
    if(initialize(nullptr)!=ALGO_RET_CHECK_PARAMS_NG||start()!=ALGO_RET_NOT_INIT||release()!=ALGO_RET_NOT_INIT)return 5;
    dlclose(lib);
    std::cout<<"{\"arm_loader_passed\":true,\"sdk_api_symbols\":8,\"sdk_struct_sizes\":[40,80,89,40],\"npu_initialized\":false,\"camera_accessed\":false}\n";
    return 0;
}
