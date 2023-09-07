#include "base/utils.h"
#include "core/optix_utils.h"
#include "kernel/types.h"

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

    const std::string program_ptx = load_file(filename);
    if (program_ptx.empty()) {
        std::cout << "Cannot find PTX file : " << filename << std::endl;
        return false; 
    }

    //size_t log_size = sizeof(msg_log);
    OPTIX_CHECK_MSG(optixModuleCreate(ctx,
                                      module_compile_options,
                                      pipeline_compile_options,
                                      program_ptx.c_str(),
                                      program_ptx.size(), optix_log_buf,
                                      &log_size, module),
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
                                            &log_size, pg),
                    fmt::format("Creating program group: {}", optix_log_buf));

    return true;
}

bool create_optix_ppl(
    const OptixDeviceContext ctx,
    const OptixPipelineCompileOptions& compile_options,
    const OptixPipelineLinkOptions& link_options,
    const std::vector<OptixProgramGroup>& pgs,
    OptixPipeline* ppl)
{
    OPTIX_CHECK_MSG(optixPipelineCreate(ctx, &compile_options, &link_options,
                                        pgs.data(), pgs.size(),
                                        optix_log_buf, &log_size, ppl),
                    fmt::format("Linking pipeline: {}", optix_log_buf));

    OptixStackSizes stack_sizes{};
    for (auto& pg : pgs)
        OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes, *ppl));

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, link_options.maxTraceDepth,
                                           0,
                                           0,
                                           &direct_callable_stack_size_from_traversal,
                                           &direct_callable_stack_size_from_state,
                                           &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(*ppl, direct_callable_stack_size_from_traversal,
                                          direct_callable_stack_size_from_state,
                                          continuation_stack_size,
                                          2 // maxTraversableDepth
                                          ));

    return true;
}

std::vector<GenericRecord> generate_records(const std::vector<OptixProgramGroup>& gps) {
    std::vector<GenericRecord> recs(gps.size());
    for (int i = 0; i < gps.size(); ++i)
        OPTIX_CHECK(optixSbtRecordPackHeader(gps[i], &recs[i]));
    return recs;
}