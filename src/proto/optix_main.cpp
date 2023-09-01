
#include "core/optix_utils.h"

int main(int argc, const char **argv) {
    std::string outfile;
    int w = 512;
    int h = 384;

    OptixDeviceContextOptions ctx_options{};
    ctx_options.logCallbackFunction = &context_log_cb;
    ctx_options.logCallbackLevel = 4;
    auto ctx = create_optix_ctx(&ctx_options);
}