
#include "core/optix_utils.h"
#include "kernel/types.h"

int main(int argc, const char **argv) {
    std::string outfile;
    int w = 512;
    int h = 384;

    OptixDeviceContextOptions ctx_options{};
    ctx_options.logCallbackFunction = &context_log_cb;
    ctx_options.logCallbackLevel = 4;
    auto ctx = create_optix_ctx(&ctx_options);

    OptixPipelineCompileOptions ppl_compile_options;

    OptixModuleCompileOptions mod_options;
    OptixModule mod_rg;
    load_optix_module("../src/kernel/device/optix/kernels.cu",
        ctx, &mod_options, &ppl_compile_options, &mod_rg);

    OptixProgramGroup rg_pg = nullptr;
    OptixProgramGroup miss_pg = nullptr;
    OptixProgramGroupOptions pg_options;

    OptixProgramGroupDesc rg_pg_desc {
        .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
        .raygen.module = mod_rg;
        .raygen.entryFunctionName = "__raygen__fixed";
    };
    create_optix_pg(ctx, &rg_pg_desc, 1, &pg_options, &rg_pg);

    OptixProgramGroupDesc miss_pg_desc {
        .kind = OPTIX_PROGRAM_GROUP_KIND_MISS
    };
    create_optix_pg(ctx, &miss_pg_desc, 1, &pg_options, &miss_pg);

    OptixPipeline ppl = nullptr;
    const uint32_t max_trace_depth = 0;
    std::vector<OptixProgramGroup> pgs { rg_pg, miss_pg };
    OptixPipelineLinkOptions ppl_link_options{};
    ppl_options.maxTraceDepth = max_trace_depth;
    create_optix_ppl(ctx, &ppl_compile_options, &ppl_link_options,
        pgs, &ppl);

    enum STAGE {
        RAYGEN,
        MISS
    };

    struct Pixel {
        float r, g, b;
    };

    std::array<CUdeviceptr, 2> records;
    const auto record_size = sizeof(GenericLocalRecord<Pixel>);
    for (auto& rec : records)
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&rec), record_size));

    std::array<GenericLocalRecord<Pixel>, 2> sbts;
    sbts[RAYGEN].data = {0.462f, 0.725f, 0.f};
    for (int i = 0; i < 2; ++i) {
        OPTIX_CHECK(optixSbtRecordPackHeader(pgs[i], &sbts[i]));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(records[i]),
            &sbts[i], record_size, cudaMemcpyHostToDevice));
    }

    OptixShaderBindingTable sbt{};
    sbt.raygenRecord            = records[RAGEN];
    sbt.missRecordBase          = records[MISS];
    sbt.missRecordStrideBytes   = sizeof(GenericLocalRecord<Pixel>);
    sbt.missRecordCount         = 1;

    CUDeviceptr output;
    auto buf_size = w * h * 3 * sizeof(float);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&output, buf_size)));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(output, 0, buf_size)));

    ParamsForTest params {
        .pixels = output,
        .image_width = w
    };

    CUDeviceptr param_ptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&param_ptr,
        sizeof(ParamsForTest))));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(param_ptr), &params,
        sizeof(ParamsForTest), cudaMemcpyHostToDevice));

    CUstream stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    OPTIX_CHECK(optixLaunch(ppl, stream, param_ptr, sizeof(ParamsForTest),
        &sbt, w, h, 1));

    cudaFree(param_ptr);

    // TODO : output the image

    cudaFree(reinterpret_cast<void*>(sbt.raygenRecord));
    cudaFree(reinterpret_cast<void*>(sbt.missRecordBase));
    
    optixPipelineDestroy(ppl);
    optixProgramGroupDestroy(miss_pg);
    optixProgramGroupDestroy(rg_pg);
    optixDeviceContextDestroy(ctx);

    return 0;
}