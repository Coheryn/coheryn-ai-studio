// Coheryn-authored isolated preprocessing harness. External headers and processor
// implementation remain in the separately pinned assessment checkout, under its MIT terms.
// This program contains no language-model or vision-encoder execution.
#include "mtmd-image.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

static void reference_log(ggml_log_level, const char *, void *) {}
clip_logger_state g_logger_state {reference_log,nullptr};
struct clip_ctx { clip_hparams parameters; };
const clip_hparams * clip_get_hparams(const clip_ctx * context) { return &context->parameters; }

int main(int argc,char ** argv) {
    try {
        if(argc!=7)throw std::runtime_error("profile width height input.rgb output.f32 output.json");
        const std::string profile=argv[1];const int width=std::stoi(argv[2]),height=std::stoi(argv[3]);
        if(width<=0||height<=0||width>8192||height>8192||int64_t(width)*height>16777216)throw std::runtime_error("invalid input extent");
        clip_ctx context;auto & hp=context.parameters;
        hp.patch_size=14;hp.image_size=336;
        const float mean[3]={0.48145466f,0.4578275f,0.40821073f};
        const float deviation[3]={0.26862954f,0.26130258f,0.27577711f};
        for(int c=0;c<3;c++){hp.image_mean[c]=mean[c];hp.image_std[c]=deviation[c];}
        std::vector<uint8_t> rgb(size_t(width)*height*3);
        std::ifstream input(argv[4],std::ios::binary);input.read(reinterpret_cast<char *>(rgb.data()),rgb.size());
        if(!input||input.peek()!=EOF)throw std::runtime_error("input RGB length mismatch");
        clip_image_u8 image;image.set_size({width,height},false);image.cpy_buf(rgb);
        mtmd_image_preproc_out result;
        if(profile=="llava15-pad336-v1") {
            hp.image_pad_color={122,116,104};hp.image_resize_pad=PAD_CEIL;
            result=mtmd_image_preprocessor_fixed_size(&context).preprocess(image);
        } else if(profile=="qwen2vl-still-v1") {
            hp.n_merge=2;hp.image_resize_pad=PAD_NONE;hp.image_round_half_even=true;
            // Explicit source-model processor limits, rather than this reference's defaults.
            hp.image_min_pixels=3136;hp.image_max_pixels=12845056;
            result=mtmd_image_preprocessor_dyn_size(&context).preprocess(image);
        } else throw std::runtime_error("unsupported processor profile");
        if(result.entries.size()!=1)throw std::runtime_error("unexpected image count");
        const auto & processed=result.entries.front();const auto & values=processed.get_ro_buf();
        std::ofstream output(argv[5],std::ios::binary);output.write(reinterpret_cast<const char *>(values.data()),values.size()*sizeof(float));
        std::ofstream metadata(argv[6]);metadata<<"{\"width\":"<<processed.nx()<<",\"height\":"<<processed.ny()<<",\"elements\":"<<values.size()<<"}\n";
        if(!output||!metadata)throw std::runtime_error("output write failed");
        return 0;
    } catch(const std::exception & error) {std::cerr<<error.what()<<"\n";return 1;}
}
