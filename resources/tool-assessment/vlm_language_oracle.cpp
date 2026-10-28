// Coheryn-authored adapter around the immutable staged research harness.
// Compile and execute only in the isolated assessment environment.
#include "llama.h"
static llama_context * capture_context(llama_model *, llama_context_params);
#define llama_init_from_model capture_context
#define main staged_capture_main
#include "vlm_oracle.cpp"
#undef main
#undef llama_init_from_model

static int language_batch = 0;
static bool capture_language(ggml_tensor * tensor, bool ask, void *) {
    const std::string name = tensor->name;
    const bool selected = name.rfind("l_out-", 0) == 0 && ggml_nelements(tensor) > 0;
    if (ask) return selected;
    if (!selected) return true;
    require(tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor), "unexpected language checkpoint layout");
    const size_t width = tensor->ne[0];
    const size_t count = ggml_nelements(tensor);
    require(width > 0 && width <= 8192 && count >= width && count <= width * 32, "language checkpoint bounds exceeded");
    if (name == "l_out-0") ++language_batch;
    std::vector<float> row(width);
    ggml_backend_tensor_get(tensor, row.data(), (count-width)*4, width*4);
    record("batch_"+std::to_string(language_batch)+"_"+name, row.data(), width, {int64_t(width)});
    return true;
}
static llama_context * capture_context(llama_model * model, llama_context_params parameters) {
    parameters.cb_eval = capture_language;
    parameters.cb_eval_user_data = nullptr;
    return llama_init_from_model(model, parameters);
}
int main(int argc, char ** argv) {
    if (argc != 6 || std::string(argv[1]) != "decode") {
        std::cerr << "decode LANGUAGE OUTPUT_DIRECTORY CONTEXT MAX_TOKENS\n";
        return 1;
    }
    return staged_capture_main(argc, argv);
}
