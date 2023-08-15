#include "base/utils.h"
#include "core/optix_utils.h"

OptixDeviceContext create_optix_ctx(const OptixDeviceContextOptions* options) {
    cudaFree(0);

    // Ensure there's at least one CUDA device
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices == 0)
        throw std::runtime_error("No available CUDA device");

    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions ctx_options{};
    ctx_options.logCallbackFunction = &context_log_cb;
    // TODO : Make this field an parameter
    ctx_options.logCallbackLevel = 4;

    // Take the current context
    CUcontext cu_ctx = 0;
    OptixDeviceContext optix_ctx;
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &ctx_options, &optix_ctx));
    return optix_ctx;
}

void destroy_optix_ctx(const OptixDeviceContext optix_ctx) {
    if (optix_ctx)
        OPTIX_CHECK(optixDeviceContextDestroy(optix_ctx));
}

bool load_optix_module(
    const char* filename,
    const OptixDeviceContext ctx,
    const OptixModuleCompileOptions* module_compile_options,
    const OptixPipelineCompileOptions* pipeline_compile_options,
    OptixModule* module)
{
    //char msg_log[8192];

    const std::string program_ptx = load_file(file_name);
    if (program_ptx.empty()) {
        std::cout << "Cannot find PTX file : " << file_name << std::endl;
        return false; 
    }

    //size_t log_size = sizeof(msg_log);
    OPTIX_CHECK_MSG(optixModuleCreate(ctx,
                                      module_compile_options,
                                      pipeline_compile_options,
                                      program_ptx.c_str(),
                                      program_ptx.size(), optix_log_buf,
                                      &optix_log_buf_size, module),
                    fmt::format("Creating module from PTX-file {}", optix_log_buf));

    return true;
}

bool create_optix_pg(
    const OptixDeviceContext ctx,
    const OptixProgramGroupDesc* pg_desc,
    const int num_pg,
    OptixProgramGroupOptions* pg_options,
    OptixProgramGroup* pg)
{
    OPTIX_CHECK_MSG(optixProgramGroupCreate(ctx, pg_desc, num_pg,
                                            pg_options, optix_log_buf,
                                            &optix_log_buf_size, pg),
                    fmt::format("Creating program group: {}", optix_log_buf));

    return true;
}