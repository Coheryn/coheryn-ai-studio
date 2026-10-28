// Coheryn-authored staged research harness. External execution remains isolated.
#include "llama.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "mtmd-helper-common.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>

static std::string directory;
static int vision_index=0;
static void require(bool ok,const char * why){if(!ok)throw std::runtime_error(why);}
static std::string json_quote(const std::string & s){std::ostringstream o;o<<'"';for(unsigned char c:s){if(c=='"'||c=='\\')o<<'\\'<<c;else if(c<32)o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else o<<c;}o<<'"';return o.str();}
static void record(const std::string & name,const float * values,size_t count,std::vector<int64_t> dims) {
    require(count>0,"empty tensor");std::ofstream raw(directory+"/"+name+".f32",std::ios::binary);raw.write(reinterpret_cast<const char *>(values),count*4);require(bool(raw),"tensor write failed");
    double square=0,sum=0;float lo=INFINITY,hi=-INFINITY;for(size_t i=0;i<count;i++){float v=values[i];require(std::isfinite(v),"nonfinite tensor");sum+=v;square+=double(v)*v;lo=std::min(lo,v);hi=std::max(hi,v);}
    std::cout<<std::setprecision(10)<<"{\"name\":"<<json_quote(name)<<",\"count\":"<<count<<",\"shape\":[";for(size_t i=0;i<dims.size();i++){if(i)std::cout<<',';std::cout<<dims[i];}std::cout<<"],\"sum\":"<<sum<<",\"l2\":"<<std::sqrt(square)<<",\"minimum\":"<<lo<<",\"maximum\":"<<hi<<",\"samples\":[";
    std::set<size_t> indices {0,std::min(size_t(1),count-1),count/2,count-1};for(size_t i=0;i<count;i+=std::max<size_t>(1,count/128))indices.insert(i);bool first=true;for(auto i:indices){if(!first)std::cout<<',';first=false;std::cout<<"{\"index\":"<<i<<",\"value\":"<<values[i]<<'}';}std::cout<<"]}"<<std::endl;
}
static bool capture(ggml_tensor * tensor,bool ask,void *) {
    std::string n=tensor->name;bool selected=n=="pre_ln"||n.rfind("layer_out-",0)==0;
    if(ask)return selected;if(!selected)return true;require(tensor->type==GGML_TYPE_F32&&ggml_is_contiguous(tensor),"unexpected checkpoint format");
    std::vector<float> values(ggml_nelements(tensor));ggml_backend_tensor_get(tensor,values.data(),0,values.size()*4);record("vision_"+std::to_string(vision_index)+"_"+n,values.data(),values.size(),{tensor->ne[0],tensor->ne[1],tensor->ne[2],tensor->ne[3]});return true;
}
template<class T>static void put(std::ostream & out,const T & v){out.write(reinterpret_cast<const char *>(&v),sizeof(v));require(bool(out),"stage write failed");}
template<class T>static T get(std::istream & in){T v;in.read(reinterpret_cast<char *>(&v),sizeof(v));require(bool(in),"truncated stage");return v;}
static void encode(const char * language,const char * projector,const char * prompt_path,const char * images_path) {
    auto mp=llama_model_default_params();mp.no_alloc=true;mp.n_gpu_layers=0;mp.load_mode=LLAMA_LOAD_MODE_NONE;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(language,mp),llama_model_free);require(bool(model),"vocabulary load failed");
    int hidden=llama_model_n_embd_inp(model.get());require(hidden>0&&hidden<=4096,"hidden bound exceeded");
    auto params=mtmd_context_params_default();params.use_gpu=false;params.n_threads=1;params.warmup=false;params.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;params.image_min_tokens=4;params.image_max_tokens=16384;params.cb_eval=capture;
    std::unique_ptr<mtmd_context,decltype(&mtmd_free)> mtmd(mtmd_init_from_file(projector,model.get(),params),mtmd_free);require(bool(mtmd),"vision initialization failed");
    std::ifstream prompt_file(prompt_path);std::string prompt((std::istreambuf_iterator<char>(prompt_file)),{});require(prompt_file.good()||prompt_file.eof(),"prompt read failed");require(prompt.size()>0&&prompt.size()<=65536,"prompt bound exceeded");
    std::ifstream image_list(images_path);std::vector<mtmd_bitmap *> images;std::vector<std::vector<unsigned char>> pixels;uint32_t w,h;std::string path;
    while(image_list>>w>>h>>std::quoted(path)){require(images.size()<2&&w>0&&h>0&&uint64_t(w)*h<=112*112,"image bound exceeded");pixels.emplace_back(size_t(w)*h*3);std::ifstream f(path,std::ios::binary);f.read(reinterpret_cast<char *>(pixels.back().data()),pixels.back().size());require(bool(f)&&f.peek()==EOF,"canonical RGB mismatch");images.push_back(mtmd_bitmap_init(w,h,pixels.back().data()));require(images.back()!=nullptr,"bitmap failed");}
    require(!images.empty(),"no images");
    std::unique_ptr<mtmd_input_chunks,decltype(&mtmd_input_chunks_free)> chunks(mtmd_input_chunks_init(),mtmd_input_chunks_free);
    mtmd_input_text text {prompt.data(),prompt.size(),true,true};require(mtmd_tokenize(mtmd.get(),chunks.get(),&text,images.data(),images.size())==0,"multimodal tokenization failed");for(auto * image:images)mtmd_bitmap_free(image);
    uint32_t count=mtmd_input_chunks_size(chunks.get());require(count>0&&count<=128,"chunk bound exceeded");uint32_t mrope=mtmd_decode_use_mrope(mtmd.get());
    std::ofstream stage(directory+"/chunks.dat",std::ios::binary);stage.write("COHERYNREF01\n",13);put(stage,count);put(stage,uint32_t(hidden));put(stage,mrope);
    llama_pos next=0;size_t physical=0;vision_index=0;
    std::cout<<"{\"stage\":\"encode\",\"hidden\":"<<hidden<<",\"mrope\":"<<(mrope?"true":"false")<<",\"marker_prompt\":"<<json_quote(prompt)<<",\"chunks\":"<<count<<"}"<<std::endl;
    for(uint32_t c=0;c<count;c++){
        const auto * chunk=mtmd_input_chunks_get(chunks.get(),c);uint32_t kind=mtmd_input_chunk_get_type(chunk),n=mtmd_input_chunk_get_n_tokens(chunk),npos=mtmd_input_chunk_get_n_pos(chunk),noncausal=mtmd_decode_use_non_causal(mtmd.get(),chunk);
        require(n>0&&n<=2048&&!noncausal,"unsupported bounded chunk");put(stage,kind);put(stage,n);put(stage,npos);put(stage,noncausal);physical+=n;require(physical<=2048,"sequence token ceiling exceeded");
        std::cout<<"{\"chunk\":"<<c<<",\"type\":"<<kind<<",\"tokens\":"<<n<<",\"position_count\":"<<npos<<",\"start_position\":"<<next;
        if(kind==MTMD_INPUT_CHUNK_TYPE_TEXT){size_t nt=0;auto * tokens=mtmd_input_chunk_get_tokens_text(chunk,&nt);require(nt==n,"text token mismatch");stage.write(reinterpret_cast<const char *>(tokens),n*4);std::cout<<",\"token_ids\":[";for(uint32_t i=0;i<n;i++){if(i)std::cout<<',';std::cout<<tokens[i];}std::cout<<"]}"<<std::endl;}
        else {require(kind==MTMD_INPUT_CHUNK_TYPE_IMAGE,"non-image media unsupported");std::cout<<",\"image_index\":"<<vision_index<<"}"<<std::endl;require(mtmd_encode_chunk(mtmd.get(),chunk)==0,"vision encoding failed");const float * embeddings=mtmd_get_output_embd(mtmd.get());require(embeddings!=nullptr,"missing embeddings");record("image_"+std::to_string(vision_index)+"_embeddings",embeddings,size_t(n)*hidden,{hidden,n,1,1});stage.write(reinterpret_cast<const char *>(embeddings),size_t(n)*hidden*4);
            auto * image=mtmd_input_chunk_get_tokens_image(chunk);std::cout<<"{\"image_positions\":"<<vision_index<<",\"positions\":[";
            for(uint32_t i=0;i<n;i++){auto pos=mtmd_image_tokens_get_decoder_pos(image,next,i);put(stage,pos.t);put(stage,pos.x);put(stage,pos.y);put(stage,pos.z);if(i)std::cout<<',';std::cout<<'['<<pos.t<<','<<pos.x<<','<<pos.y<<','<<pos.z<<']';}std::cout<<"]}"<<std::endl;vision_index++;
        }
        next+=npos;
    }
    require(bool(stage),"stage output failed");std::cout<<"{\"stage\":\"encoded\",\"physical_prompt_tokens\":"<<physical<<",\"effective_positions\":"<<next<<",\"images\":"<<vision_index<<"}"<<std::endl;
}
static void decode(const char * language,int context,int maximum) {
    require(context>=128&&context<=2048&&maximum>=1&&maximum<=8,"decode bounds exceeded");
    auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.load_mode=LLAMA_LOAD_MODE_MMAP;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(language,mp),llama_model_free);require(bool(model),"language model load failed");
    auto cp=llama_context_default_params();cp.n_ctx=context;cp.n_batch=32;cp.n_ubatch=32;cp.n_threads=1;cp.n_threads_batch=1;cp.type_k=GGML_TYPE_F32;cp.type_v=GGML_TYPE_F32;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=false;cp.op_offload=false;
    std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);require(bool(ctx),"language context failed");
    std::ifstream stage(directory+"/chunks.dat",std::ios::binary);char magic[13];stage.read(magic,13);require(std::string(magic,13)=="COHERYNREF01\n","stage identity mismatch");uint32_t count=get<uint32_t>(stage),hidden=get<uint32_t>(stage),mrope=get<uint32_t>(stage);require(count>0&&count<=128&&hidden==uint32_t(llama_model_n_embd_inp(model.get()))&&mrope<=1,"stage bounds mismatch");llama_pos next=0;size_t physical=0;
    for(uint32_t c=0;c<count;c++){
        uint32_t kind=get<uint32_t>(stage),n=get<uint32_t>(stage),npos=get<uint32_t>(stage),noncausal=get<uint32_t>(stage);require(n>0&&n<=2048&&npos>0&&!noncausal,"invalid stage chunk");physical+=n;require(physical+maximum<=uint32_t(context),"context exceeded");
        if(kind==MTMD_INPUT_CHUNK_TYPE_TEXT){std::vector<llama_token> tokens(n);stage.read(reinterpret_cast<char *>(tokens.data()),n*4);require(bool(stage),"text stage truncated");
            llama_batch_ext_ptr batch(llama_batch_ext_init(ctx.get()));for(uint32_t offset=0;offset<n;offset+=32){llama_batch_ext_clear(batch.get());for(uint32_t i=offset;i<std::min(n,offset+32);i++){int id=llama_batch_ext_add_token(batch.get(),0,tokens[i]);require(id>=0,"text batch failed");llama_pos pos=next+i;require(llama_batch_ext_set_pos(batch.get(),id,&pos),"text position failed");if(c+1==count&&i+1==n)llama_batch_ext_set_output_logits(batch.get(),id,true);}require(llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get())==0,"text decode failed");}}
        else {require(kind==MTMD_INPUT_CHUNK_TYPE_IMAGE,"unknown staged media");std::vector<float> emb(size_t(n)*hidden);stage.read(reinterpret_cast<char *>(emb.data()),emb.size()*4);std::vector<mtmd_decoder_pos> positions(n);for(auto & p:positions){p.t=get<uint32_t>(stage);p.x=get<uint32_t>(stage);p.y=get<uint32_t>(stage);p.z=get<uint32_t>(stage);}decode_embd_batch batch(emb.data(),n,mrope?4:1,hidden);if(mrope)batch.set_position_mrope_2d(positions,0);else batch.set_position_normal(next,0);for(uint32_t offset=0;offset<n;offset+=32){require(llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.render(ctx.get(),offset,std::min(n-offset,32u)))==0,"image embedding decode failed");std::cerr<<"image prefill "<<offset+std::min(n-offset,32u)<<"/"<<n<<std::endl;}}
        next+=npos;
    }
    require(stage.peek()==EOF,"stage trailing bytes");auto * vocab=llama_model_get_vocab(model.get());int size=llama_vocab_n_tokens(vocab);require(size>0&&size<=200000,"vocabulary ceiling exceeded");std::vector<llama_token> generated;
    for(int step=0;step<maximum;step++){
        const float * logits=llama_get_logits_ith(ctx.get(),-1);require(logits!=nullptr,"missing language logits");record("logits_"+std::to_string(step),logits,size,{size});llama_token id=std::max_element(logits,logits+size)-logits;generated.push_back(id);std::vector<char> piece(1024);int piece_size=llama_token_to_piece(vocab,id,piece.data(),piece.size(),0,false);require(piece_size>=0&&piece_size<=1024,"token piece exceeds bound");std::ostringstream hex;for(int i=0;i<piece_size;i++)hex<<std::hex<<std::setw(2)<<std::setfill('0')<<int(static_cast<unsigned char>(piece[i]));std::cout<<"{\"greedy_step\":"<<step<<",\"token_id\":"<<id<<",\"piece_hex\":"<<json_quote(hex.str())<<",\"eos\":"<<(llama_vocab_is_eog(vocab,id)?"true":"false")<<"}"<<std::endl;if(llama_vocab_is_eog(vocab,id)||step+1==maximum)break;
        llama_batch_ext_ptr batch(llama_batch_ext_init(ctx.get()));int row=llama_batch_ext_add_token(batch.get(),0,id);require(row>=0,"decode token batch failed");require(llama_batch_ext_set_pos(batch.get(),row,&next),"generation position failed");llama_batch_ext_set_output_logits(batch.get(),row,true);require(llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get())==0,"generation decode failed");next++;
    }
    std::cout<<"{\"stage\":\"decoded\",\"context\":"<<context<<",\"physical_prompt_tokens\":"<<physical<<",\"generated_token_ids\":[";for(size_t i=0;i<generated.size();i++){if(i)std::cout<<',';std::cout<<generated[i];}std::cout<<"]}"<<std::endl;
}
int main(int argc,char ** argv){try{require(argc==7||argc==6,"encode MODEL PROJECTOR PROMPT IMAGE_LIST OUTPUT or decode MODEL OUTPUT CONTEXT MAXTOKENS");ggml_backend_load_all();llama_backend_init();std::string mode=argv[1];if(mode=="encode"){directory=argv[6];encode(argv[2],argv[3],argv[4],argv[5]);}else if(mode=="decode"){directory=argv[3];decode(argv[2],std::stoi(argv[4]),std::stoi(argv[5]));}else throw std::runtime_error("unknown stage");llama_backend_free();struct rusage usage{};require(getrusage(RUSAGE_SELF,&usage)==0,"rusage unavailable");
#ifdef __APPLE__
        uint64_t peak=usage.ru_maxrss;
#else
        uint64_t peak=uint64_t(usage.ru_maxrss)*1024;
#endif
        std::cout<<"{\"process_peak_rss_bytes\":"<<peak<<"}"<<std::endl;return 0;}catch(const std::exception & e){std::cerr<<e.what()<<std::endl;return 1;}}
