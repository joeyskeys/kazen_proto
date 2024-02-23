#include <OpenImageIO/imageio.h>
#include <OpenImageIO/argparse.h>

#include <optix_function_table_definition.h>

#include "core/optix_utils.h"
#include "kernel/types.h"

using RaygenRecord   = GenericLocalRecord<RaygenData>;
using MissRecord     = GenericLocalRecord<MissData>;
using HitGroupRecord = GenericLocalRecord<HitGroupData>;

int main(int argc, const char **argv) {
    std::string outfile;
    OIIO::ArgParse ap;
    bool debug = false;
    uint32_t w = 512;
    uint32_t h = 384;
    int debug_x = -1;
    int debug_y = -1;

    ap.intro("Kazen Render GPU")
        .usage("kazen_gpu [options] filename")
        .print_defaults(true);

    ap.separator("Options:");
    ap.arg("-o %s", &outfile)
        .help("output filename");

    ap.arg("-d", &debug)
        .help("enable debug mode");

    ap.arg("--pixel %d %d", &debug_x, &debug_y)
        .help("specify the pixel of interest");

    if (ap.parse(argc, argv) < 0) {
        std::cerr << ap.geterror() << std::endl;
        ap.print_help();
        return 0;
    }

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
    OptixProgramGroup ch_pg = nullptr;
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
            .entryFunctionNameCH = "__closesthit_radiance"
        }
    };
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
    size_t hg_record_size = sizeof(HitGroupRecord);
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