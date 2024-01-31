#include <OpenImageIO/imageio.h>
#include <optix_function_table_definition.h>

#include "core/optix_utils.h"
#include "kernel/types.h"

using GenericLocalRecord<RaygenData>    RaygenRecord;
using GenericLocalRecord<MissData>      MissRecord;
using GenericLocalRecord<HitGroupData>  HitGroupRecord;

int main(int argc, const char **argv) {
    std::string outfile;
    uint32_t w = 512;
    uint32_t h = 384;

    OptixDeviceContextOptions ctx_options{};
    ctx_options.logCallbackFunction = &context_log_cb;
    ctx_options.logCallbackLevel = 4;
    auto ctx = create_optix_ctx(&ctx_options);

    OptixModuleCompileOptions mod_options{
        .optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
        .debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL
    };
    OptixPipelineCompileOptions ppl_compile_options {
        .usesMotionBlur = false,
        .numPayloadValues = 2,
        .numAttributeValues = 2,
        .exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
        .pipelineLaunchParamsVariableName = "params"
    };
    OptixModule mod;
    load_optix_module_cu("../src/kernel/device/optix/kernels.cu",
        ctx, &mod_options, &ppl_compile_options, &mod);

    OptixProgramGroup rg_pg = nullptr;
    OptixProgramGroup miss_pg = nullptr;
    OpitxProgramGroup ch_pg = nullptr;
    OptixProgramGroupOptions pg_options = {};

    OptixProgramGroupDesc rg_pg_desc {
        .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
        .raygen = {
            .module = mod,
            .entryFunctionName = "__raygen__main"
        }
    };
    create_optix_pg(ctx, &rg_pg_desc, 1, &pg_options, &rg_pg);

    OptixProgramGroupDesc miss_pg_desc {
        .kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
        .miss = {
            .module = mod,
            .entryFunctionName = "__miss_radiance"
        }
    };
    create_optix_pg(ctx, &miss_pg_desc, 1, &pg_options, &miss_pg);

    OptixProgramGroupDesc ch_pg_desc {
        .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        .hitgroup = {
            .moduleCH = mod,
            .entryFunctionName = "__closesthit_radiance"
        }
    }
    create_optix_pg(ctx, &ch_pg_desc, 1, &pg_options, &ch_pg);

    OptixPipeline ppl = nullptr;
    const uint32_t max_trace_depth = 2;
    //std::vector<OptixProgramGroup> pgs { rg_pg, miss_pg };
    std::vector<OptixProgramGroup> pgs { rg_pg, miss_pg, ch_pg };
    OptixPipelineLinkOptions ppl_link_options{};
    ppl_link_options.maxTraceDepth = max_trace_depth;
    create_optix_ppl(ctx, ppl_compile_options, ppl_link_options,
        pgs, &ppl);

    struct Pixel {
        float r, g, b;
    };

    std::array<CUdeviceptr, 3> records;
    /*
    const auto record_size = sizeof(GenericLocalRecord<Pixel>);
    const size_t rec_sizes[] = { sizeof(raygenRecord), sizeof(MissRecord), sizeof(HitGroupRecord) };
    for (auto& rec : records)
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&rec), record_size));

    std::array<GenericLocalRecord<Pixel>, 2> sbts;
    sbts[RAYGEN].data = {0.462f, 0.725f, 0.f};
    for (int i = 0; i < 2; ++i) {
        OPTIX_CHECK(optixSbtRecordPackHeader(pgs[i], &sbts[i]));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(records[i]),
            &sbts[i], record_size, cudaMemcpyHostToDevice));
    }
    */

    const size_t rg_record_size = sizeof(RaygenRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&records[RAYGEN]), rg_record_size));
    RaygenRecord rg_sbt = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(rg_pg, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(records[RAYGEN]),
        &rg_sbt, rg_record_size, cudaMemcpyHostToDevice));

    const size_t ms_record_size = sizeof(MissRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&records[MISS]), ms_record_size * RAY_TYPE_COUNT));
    MissRecord ms_sbt[1];
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg, &ms_sbt[0]));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(records[MISS]), ms_sbt,
        ms_record_size * RAY_TYPE_COUNT, cudaMemcpyHostToDevice));

    const size_t MAT_COUNT = 1;
    HitGroupRecord hg_record_size = sizeof(HitGroupRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&records[CLOSESTHIT]),
        hg_record_size * RAY_TYPE_COUNT * MAT_COUNT));
    HitGroupRecord hg_records[RAY_TYPE_COUNT * MAT_COUNT];
    for (int i = 0; i < MAT_COUNT; ++i) {
        const int sbt_idx = i * RAY_TYPE_COUNT + 0;
        OPTIX_CHECK(optixSbtRecordPackHeader(ch_pg, &hg_records[sbt_idx]));
        hg_records[sbt_idx].data.emission_color = g_emission_colors[i];
        hg_records[sbt_idx].data.diffuse_color = g_diffuse_colors[i];
        hg_records[sbt_idx].data.vertices = reinterpret_cast<float3*>();
    }
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(records[CLOSESTHIT]), hg_records,
        hg_record_size * RAY_TYPE_COUNT * MAT_COUNT, cudaMemcpyHostToDevice));

    OptixShaderBindingTable sbt{};
    sbt.raygenRecord                = records[RAYGEN];
    sbt.missRecordBase              = records[MISS];
    sbt.missRecordStrideInBytes     = static_cast<uint32_t>(ms_record_size);
    sbt.missRecordCount             = RAY_TYPE_COUNT;
    sbt.hitgroupRecordBase          = records[CLOSESTHIT];
    sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hg_record_size);
    sbt.hitgroupRecordCount         = RAY_TYPE_COUNT * MAT_COUNT;

    /*
    OptixShaderBindingTable sbt{};
    {
        CUdeviceptr rg_record;
        const size_t rg_record_size = sizeof(GenericLocalRecord<Pixel>);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&rg_record), rg_record_size));
        GenericLocalRecord<Pixel> rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(rg_pg, &rg_sbt));
        rg_sbt.data = {0.462f, 0.725f, 0.f};
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(rg_record),
            &rg_sbt,
            rg_record_size,
            cudaMemcpyHostToDevice
        ));

        CUdeviceptr ms_record;
        const size_t ms_record_size = sizeof(GenericLocalRecord<int>);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ms_record), ms_record_size));
        GenericLocalRecord<int> ms_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg, &ms_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(ms_record),
            &ms_sbt,
            ms_record_size,
            cudaMemcpyHostToDevice
        ));

        sbt.raygenRecord = rg_record;
        sbt.missRecordBase = ms_record;
        sbt.missRecordStrideInBytes = sizeof(GenericLocalRecord<int>);
        sbt.missRecordCount = 1;
    }
    */

    CUdeviceptr output;
    auto buf_size = w * h * 3 *sizeof(float);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&output), buf_size));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(output), 0, buf_size));

    CUstream stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    ParamsForTest params {
        .image = output,
        .image_width = w
    };

    CUdeviceptr param_ptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&param_ptr),
        sizeof(ParamsForTest)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(param_ptr), &params,
        sizeof(ParamsForTest), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(ppl, stream, param_ptr, sizeof(ParamsForTest),
        &sbt, w, h, 1));

    // Output the image
    std::vector<float> host_data(w * h * 3);
    CUDA_CHECK(cudaMemcpy(host_data.data(), reinterpret_cast<void*>(output),
        buf_size, cudaMemcpyDeviceToHost));
    auto spec = OIIO::ImageSpec(w, h, 3, OIIO::TypeDesc::UINT8);
    auto oiio_out = OIIO::ImageOutput::create("test.png");
    oiio_out->open("test.png", spec);
    oiio_out->write_image(OIIO::TypeDesc::FLOAT, host_data.data());

    cudaFree(reinterpret_cast<void*>(sbt.raygenRecord));
    cudaFree(reinterpret_cast<void*>(sbt.missRecordBase));
    cudaFree(reinterpret_cast<void*>(param_ptr));
    cudaFree(reinterpret_cast<void*>(output));
    
    optixPipelineDestroy(ppl);
    optixProgramGroupDestroy(miss_pg);
    optixProgramGroupDestroy(rg_pg);
    optixDeviceContextDestroy(ctx);

    return 0;
}