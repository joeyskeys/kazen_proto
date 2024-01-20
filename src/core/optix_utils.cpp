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

std::string cu_to_ptx(const char* inc_dir, const char* cu_str, const char* name,
    const std::vector<const char*>& compiler_options)
{
    // Create program
    nvrtcProgram prog = 0;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, cu_str, name, 0, NULL, NULL));

    // Gather options
    std::vector<const char*> options;
    if (inc_dir) {
        auto inc_option = std::string("-I") + inc_dir;
        options.push_back(inc_option.c_str());
    }

    std::copy(std::begin(compiler_options), std::end(compiler_options), std::back_inserter(options));

    // JIT compile CU to PTX, not considering OPTIXIR for now
    const nvrtcResult ret = nvrtcCompileProgram(prog, (int)options.size(), options.data());

    // Retrieve log output
    NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &log_size));
    log_size = std::min(static_cast<int>(log_size), LOG_BUF_SIZE - 1);
    if (log_size > 1)
        NVRTC_CHECK(nvrtcGetProgramLog(prog, log_buf));
    if (ret != NVRTC_SUCCESS) {
        fmt::print(stderr, "nvrtc compile error: {}", log_buf);
        exit(1);
    }

    // Retrieve PTX code
    size_t ptx_size = 0;
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));
    std::string ptx;
    ptx.resize(ptx_size);
    nvrtcGetPTX(prog, ptx.data());

    // Cleanup
    nvrtcDestroyProgram(&prog);

    return ptx;
}

bool load_optix_module_ptx(
    const char* filename,
    const OptixDeviceContext ctx,
    const OptixModuleCompileOptions* module_compile_options,
    const OptixPipelineCompileOptions* pipeline_compile_options,
    OptixModule* module)
{
    const std::string program_ptx = load_file(filename);
    if (program_ptx.empty()) {
        std::cout << "Cannot find PTX file : " << filename << std::endl;
        return false; 
    }

    OPTIX_CHECK_MSG(optixModuleCreate(ctx,
                                      module_compile_options,
                                      pipeline_compile_options,
                                      program_ptx.c_str(),
                                      program_ptx.size(), log_buf,
                                      &log_size, module),
                    fmt::format("Creating module from PTX-file {}", log_buf));

    return true;
}

bool load_optix_module_cu(
    const char* filename,
    const OptixDeviceContext ctx,
    const OptixModuleCompileOptions* module_compile_options,
    const OptixPipelineCompileOptions* pipeline_compile_options,
    OptixModule* module)
{
    const std::string program_cu = load_file(filename);
    if (program_cu.empty()) {
        std::cout << "Cannot find cu file : " << filename << std::endl;
        return false; 
    }
    const std::string program_ptx = cu_to_ptx(nullptr, program_cu.c_str(),
        filename);

    OPTIX_CHECK_MSG(optixModuleCreate(ctx,
                                      module_compile_options,
                                      pipeline_compile_options,
                                      program_ptx.c_str(),
                                      program_ptx.size(), log_buf,
                                      &log_size, module),
                    fmt::format("Creating module from PTX-file {}", log_buf));

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
                                            pg_options, log_buf,
                                            &log_size, pg),
                    fmt::format("Creating program group: {}", log_buf));

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
                                        log_buf, &log_size, ppl),
                    fmt::format("Linking pipeline: {}", log_buf));

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