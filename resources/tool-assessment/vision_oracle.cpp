// Coheryn-authored independent vision capture, kept outside the native build/runtime.
#include "clip.h"
#include "clip-model.h"
#include "mtmd-image.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>

struct capture_state {std::string directory;};
static void record(const std::string & name,const std::vector<float> & values,const int64_t * dims,const std::string & directory) {
    const std::string path=directory+"/"+name+".f32";
    std::ofstream raw(path,std::ios::binary);raw.write(reinterpret_cast<const char *>(values.data()),values.size()*4);
    if(!raw)throw std::runtime_error("raw tensor write failed");
    double square=0,sum=0;float minimum=INFINITY,maximum=-INFINITY;
    for(auto v:values){if(!std::isfinite(v))throw std::runtime_error("nonfinite reference output");sum+=v;square+=double(v)*v;minimum=std::min(minimum,v);maximum=std::max(maximum,v);}
    std::cout<<std::setprecision(10)<<"{\"name\":\""<<name<<"\",\"shape\":["<<dims[0]<<","<<dims[1]<<","<<dims[2]<<","<<dims[3]<<"],\"count\":"<<values.size()<<",\"sum\":"<<sum<<",\"l2\":"<<std::sqrt(square)<<",\"minimum\":"<<minimum<<",\"maximum\":"<<maximum<<",\"samples\":[";
    std::set<size_t> indices {0,1,values.size()/2,values.size()-1};
    for(size_t i=0;i<values.size();i+=std::max<size_t>(1,values.size()/128))indices.insert(i);
    bool first=true;for(auto index:indices){if(!first)std::cout<<',';first=false;std::cout<<"{\"index\":"<<index<<",\"value\":"<<values[index]<<'}';}
    std::cout<<"]}"<<std::endl;
}
static bool inspect_tensor(ggml_tensor * tensor,bool ask,void * user) {
    const std::string name=tensor->name;
    bool selected=name=="pre_ln" || name.rfind("layer_out-",0)==0;
    if(ask)return selected;
    if(!selected)return true;
    if(tensor->type!=GGML_TYPE_F32 || !ggml_is_contiguous(tensor))return false;
    std::vector<float> values(ggml_nelements(tensor));ggml_backend_tensor_get(tensor,values.data(),0,values.size()*4);
    record(name,values,tensor->ne,static_cast<capture_state *>(user)->directory);return true;
}
int main(int argc,char ** argv) {
    try {
        if(argc!=7)throw std::runtime_error("model profile width height canonical.rgb output-directory");
        const std::string profile=argv[2];const int width=std::stoi(argv[3]),height=std::stoi(argv[4]);
        if(width<=0||height<=0||int64_t(width)*height>112*112)throw std::runtime_error("bounded reference image required");
        capture_state capture {argv[6]};ggml_backend_load_all();
        clip_context_params params {};params.use_gpu=false;params.flash_attn_type=CLIP_FLASH_ATTN_TYPE_DISABLED;
        params.image_min_tokens=4;params.image_max_tokens=16384;params.warmup=false;params.no_alloc=false;
        params.cb_eval=inspect_tensor;params.cb_eval_user_data=&capture;
        auto result=clip_init(argv[1],params);auto * context=result.ctx_v;
        if(!context)throw std::runtime_error("reference vision component load failed");
        std::vector<uint8_t> rgb(size_t(width)*height*3);
        std::ifstream input(argv[5],std::ios::binary);input.read(reinterpret_cast<char *>(rgb.data()),rgb.size());
        if(!input||input.peek()!=EOF)throw std::runtime_error("canonical RGB byte count mismatch");
        clip_image_u8 image;image.set_size({width,height},false);image.cpy_buf(rgb);
        auto prepared=profile=="llava15-pad336-v1" ? mtmd_image_preprocessor_fixed_size(context).preprocess(image) : mtmd_image_preprocessor_dyn_size(context).preprocess(image);
        if(prepared.entries.size()!=1)throw std::runtime_error("unexpected reference preprocessing result");
        const auto & processed=prepared.entries.front();
        const auto * hp=clip_get_hparams(context);
        std::cout<<"{\"hparams\":{\"layers\":"<<hp->n_layer<<",\"hidden\":"<<hp->n_embd<<",\"patch\":"<<hp->patch_size<<",\"min_pixels\":"<<hp->image_min_pixels<<",\"max_pixels\":"<<hp->image_max_pixels<<"},\"processed_size\":["<<processed.nx()<<","<<processed.ny()<<"]}"<<std::endl;
        std::vector<float> encoded(size_t(clip_n_mmproj_embd(context))*clip_n_output_tokens(context,&processed));
        if(!clip_image_encode(context,1,&processed,encoded))throw std::runtime_error("reference vision execution failed");
        int64_t dims[4] {clip_n_mmproj_embd(context),clip_n_output_tokens(context,&processed),1,1};
        if(encoded.size()!=size_t(dims[0])*dims[1])throw std::runtime_error("projected embedding shape mismatch");
        record("projected_embeddings",encoded,dims,capture.directory);
        clip_free(context);return 0;
    }catch(const std::exception & error){std::cerr<<error.what()<<"\n";return 1;}
}
