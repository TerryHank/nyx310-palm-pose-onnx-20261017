// Host-only API transport. Never included in the ARM upload package.
#include "rknn_api.h"
#include <vector>
#include <map>
#include <cstring>
#include <string>
#include <cstdint>
using Infer=int(*)(int,const uint8_t*,int,float**,const uint32_t*);
static Infer inferCallback=nullptr;
struct Context {int kind,size;std::vector<uint32_t> counts;const uint8_t* input=nullptr;std::vector<std::vector<float>> output;};
static std::map<rknn_context,Context> contexts;
static rknn_context nextContext=1;
static int fail=0;
extern "C" void mock_set_callback(Infer cb){inferCallback=cb;}
extern "C" void mock_fail_next(int operation){fail=operation;}
extern "C" int mock_context_count(){return int(contexts.size());}
extern "C" int rknn_init(rknn_context* ctx,void* model,uint32_t size,uint32_t,rknn_init_extend*){
    if(!ctx||!model||size!=0)return -1;
    std::string path(static_cast<char*>(model));bool palm=path.find("palm.rknn")!=std::string::npos;
    bool gesture=path.find("hagrid.rknn")!=std::string::npos;
    Context c;c.kind=gesture?2:palm?0:1;c.size=gesture?640:palm?192:224;
    c.counts=gesture?std::vector<uint32_t>{64*80*60,34*80*60,64*40*30,34*40*30,64*20*15,34*20*15}:palm?std::vector<uint32_t>{2016*18,2016}:std::vector<uint32_t>{63,1,1,63};
    *ctx=nextContext++;contexts[*ctx]=c;return 0;
}
extern "C" int rknn_destroy(rknn_context ctx){return contexts.erase(ctx)?0:-1;}
extern "C" int rknn_query(rknn_context ctx,rknn_query_cmd cmd,void* value,uint32_t){
    auto it=contexts.find(ctx);if(it==contexts.end())return -1;auto& c=it->second;
    if(cmd==RKNN_QUERY_IN_OUT_NUM){auto* v=static_cast<rknn_input_output_num*>(value);v->n_input=1;v->n_output=c.counts.size();return 0;}
    if(cmd==RKNN_QUERY_INPUT_ATTR||cmd==RKNN_QUERY_OUTPUT_ATTR){auto* v=static_cast<rknn_tensor_attr*>(value);
        v->n_elems=cmd==RKNN_QUERY_INPUT_ATTR?c.size*(c.kind==2?480:c.size)*3:c.counts.at(v->index);v->type=RKNN_TENSOR_FLOAT32;v->fmt=RKNN_TENSOR_NCHW;return 0;}
    return -1;
}
extern "C" int rknn_inputs_set(rknn_context ctx,uint32_t count,rknn_input in[]){
    if(fail==1){fail=0;return -99;}auto& c=contexts.at(ctx);
    if(count!=1||in[0].size!=uint32_t(c.size*(c.kind==2?480:c.size)*3)||in[0].fmt!=RKNN_TENSOR_NHWC||in[0].type!=RKNN_TENSOR_UINT8||in[0].pass_through)return -1;
    c.input=static_cast<uint8_t*>(in[0].buf);return 0;
}
extern "C" int rknn_run(rknn_context ctx,rknn_run_extend*){
    if(fail==2){fail=0;return -99;}auto& c=contexts.at(ctx);if(!inferCallback||!c.input)return -1;
    c.output.clear();std::vector<float*> pointers;
    for(auto count:c.counts)c.output.emplace_back(count,0);
    for(auto& output:c.output)pointers.push_back(output.data());
    return inferCallback(c.kind,c.input,c.size,pointers.data(),c.counts.data());
}
extern "C" int rknn_outputs_get(rknn_context ctx,uint32_t count,rknn_output out[],rknn_output_extend*){
    if(fail==3){fail=0;return -99;}auto& c=contexts.at(ctx);if(count!=c.counts.size())return -1;
    for(size_t i=0;i<count;i++){if(!out[i].want_float)return -1;out[i].buf=c.output[i].data();out[i].size=c.counts[i]*4;}return 0;
}
extern "C" int rknn_outputs_release(rknn_context,uint32_t,rknn_output[]){return 0;}
