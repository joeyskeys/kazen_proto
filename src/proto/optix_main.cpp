#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/argparse.h>

//#include <optix_function_table_definition.h>

#include "base/vec.h"
#include "core/accel.h"
#include "core/optix_utils.h"
#include "kernel/types.h"
#include "kernel/hostutils.h"
#include "kernel/mathutils.h"

//using RaygenRecord   = GenericLocalRecord<RaygenData>;
//using MissRecord     = GenericLocalRecord<MissData>;
//using HitGroupRecord = GenericLocalRecord<HitGroupData>;
using RaygenRecord   = GenericLocalRecord<RaygenDataTriangle>;
using MissRecord     = GenericLocalRecord<MissDataTriangle>;
using HitGroupRecord = GenericLocalRecord<HitgroupDataTriangle>;

const size_t MAT_COUNT = 1;

const std::array<float3, MAT_COUNT> g_emission_colors = {
    {0.8f, 0.8f, 0.8f}
};

const std::array<float3, MAT_COUNT> g_diffuse_colors = {
    {0.8f, 0.05f, 0.05f}
};

const unsigned int payload_semantics[18] = {
    // RadiancePRD::attenuation
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    // RadiancePRD::seed
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    // RadiancePRD::depth
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    // RadiancePRD::emitted
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    // RadiancePRD::radiance
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    // RadiancePRD::origin
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    // RadiancePRD::direction
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    // RadiancePRD::done
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE
};

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

    OptixPayloadType payload_type {
        .numPayloadValues = sizeof(payload_semantics) / sizeof(payload_semantics[0]),
        .payloadSemantics = payload_semantics
    };

    OptixModuleCompileOptions mod_options {
        .optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
        .debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL,
        .numPayloadTypes = 1,
        .payloadTypes = &payload_type
    };

    OptixModuleCompileOptions mod_options_triangle = {};

    OptixPipelineCompileOptions ppl_compile_options {
        .usesMotionBlur = false,
        .numPayloadValues = 3,
        .numAttributeValues = 3,
        .exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
        .pipelineLaunchParamsVariableName = "params",
        .usesPrimitiveTypeFlags = static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE)
    };

    OptixModule mod;
    load_optix_module_cu("../src/kernel/device/optix/kernels_triangle.cu",
        ctx, &mod_options_triangle, &ppl_compile_options, &mod);

    OptixProgramGroup rg_pg = nullptr;
    OptixProgramGroup miss_pg = nullptr;
    OptixProgramGroup ch_pg = nullptr;
    OptixProgramGroupOptions pg_options = {};

    OptixProgramGroupDesc rg_pg_desc {
        .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
        .raygen = {
            .module = mod,
            .entryFunctionName = "__raygen__rg_triangle"
        }
    };
    create_optix_pg(ctx, &rg_pg_desc, 1, &pg_options, &rg_pg);

    OptixProgramGroupDesc miss_pg_desc {
        .kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
        .miss = {
            .module = mod,
            .entryFunctionName = "__miss__ms_triangle"
        }
    };
    create_optix_pg(ctx, &miss_pg_desc, 1, &pg_options, &miss_pg);

    OptixProgramGroupDesc ch_pg_desc {
        .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        .hitgroup = {
            .moduleCH = mod,
            .entryFunctionNameCH = "__closesthit__ch_triangle"
        }
    };
    create_optix_pg(ctx, &ch_pg_desc, 1, &pg_options, &ch_pg);

    OptixPipeline ppl = nullptr;
    const uint32_t max_trace_depth = 1;
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
    ms_sbt[0].data.bg_color = {0.3f, 0.1f, 0.2f};
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg, &ms_sbt[0]));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(records[MISS]), ms_sbt,
        ms_record_size * RAY_TYPE_COUNT, cudaMemcpyHostToDevice));

    size_t hg_record_size = sizeof(HitGroupRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&records[CLOSESTHIT]),
        hg_record_size * RAY_TYPE_COUNT * MAT_COUNT));
    HitGroupRecord hg_records[RAY_TYPE_COUNT * MAT_COUNT];
    for (int i = 0; i < MAT_COUNT; ++i) {
        const int sbt_idx = i * RAY_TYPE_COUNT + 0;
        OPTIX_CHECK(optixSbtRecordPackHeader(ch_pg, &hg_records[sbt_idx]));
        /*
        hg_records[sbt_idx].data.emission_color = g_emission_colors[i];
        hg_records[sbt_idx].data.diffuse_color = g_diffuse_colors[i];
        hg_records[sbt_idx].data.vertices = nullptr;
        */
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
    auto buf_size = w * h * 4 * sizeof(float);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&output), buf_size));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(output), 0, buf_size));

    std::vector<base::Vec3f> vs = {
        Vec3f{-1.f, -1.f, -1.f}, // 0
        Vec3f{ 1.f, -1.f, -1.f}, // 1
        Vec3f{ 1.f,  1.f, -1.f}, // 2
        Vec3f{-1.f,  1.f, -1.f}, // 3
        Vec3f{-1.f, -1.f,  1.f}, // 4
        Vec3f{ 1.f, -1.f,  1.f}, // 5
        Vec3f{ 1.f,  1.f,  1.f}, // 6
        Vec3f{-1.f,  1.f,  1.f}, // 7
    };
    std::vector<base::Vec3i> idx = {
        // back
        Vec3i{0, 3, 1},
        Vec3i{1, 3, 2},
        // front
        Vec3i{4, 5, 6},
        Vec3i{4, 6, 7},
        // left
        Vec3i{0, 4, 3},
        Vec3i{4, 7, 3},
        // right
        Vec3i{5, 1, 2},
        Vec3i{5, 2, 6},
        // bottom
        Vec3i{4, 0, 1},
        Vec3i{4, 1, 5},
        // top
        Vec3i{7, 6, 2},
        Vec3i{7, 2, 3}
    };

    std::vector<base::Vec4f> g_vertices = {
    // Floor  -- white lambert
    {    0.0f,    0.0f,    0.0f, 0.0f },
    {    0.0f,    0.0f,  559.2f, 0.0f },
    {  556.0f,    0.0f,  559.2f, 0.0f },
    {    0.0f,    0.0f,    0.0f, 0.0f },
    {  556.0f,    0.0f,  559.2f, 0.0f },
    {  556.0f,    0.0f,    0.0f, 0.0f },

    // Ceiling -- white lambert
    {    0.0f,  548.8f,    0.0f, 0.0f },
    {  556.0f,  548.8f,    0.0f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },

    {    0.0f,  548.8f,    0.0f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },
    {    0.0f,  548.8f,  559.2f, 0.0f },

    // Back wall -- white lambert
    {    0.0f,    0.0f,  559.2f, 0.0f },
    {    0.0f,  548.8f,  559.2f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },

    {    0.0f,    0.0f,  559.2f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },
    {  556.0f,    0.0f,  559.2f, 0.0f },

    // Right wall -- green lambert
    {    0.0f,    0.0f,    0.0f, 0.0f },
    {    0.0f,  548.8f,    0.0f, 0.0f },
    {    0.0f,  548.8f,  559.2f, 0.0f },

    {    0.0f,    0.0f,    0.0f, 0.0f },
    {    0.0f,  548.8f,  559.2f, 0.0f },
    {    0.0f,    0.0f,  559.2f, 0.0f },

    // Left wall -- red lambert
    {  556.0f,    0.0f,    0.0f, 0.0f },
    {  556.0f,    0.0f,  559.2f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },

    {  556.0f,    0.0f,    0.0f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },
    {  556.0f,  548.8f,    0.0f, 0.0f },

    // Short block -- white lambert
    {  130.0f,  165.0f,   65.0f, 0.0f },
    {   82.0f,  165.0f,  225.0f, 0.0f },
    {  242.0f,  165.0f,  274.0f, 0.0f },

    {  130.0f,  165.0f,   65.0f, 0.0f },
    {  242.0f,  165.0f,  274.0f, 0.0f },
    {  290.0f,  165.0f,  114.0f, 0.0f },

    {  290.0f,    0.0f,  114.0f, 0.0f },
    {  290.0f,  165.0f,  114.0f, 0.0f },
    {  240.0f,  165.0f,  272.0f, 0.0f },

    {  290.0f,    0.0f,  114.0f, 0.0f },
    {  240.0f,  165.0f,  272.0f, 0.0f },
    {  240.0f,    0.0f,  272.0f, 0.0f },

    {  130.0f,    0.0f,   65.0f, 0.0f },
    {  130.0f,  165.0f,   65.0f, 0.0f },
    {  290.0f,  165.0f,  114.0f, 0.0f },

    {  130.0f,    0.0f,   65.0f, 0.0f },
    {  290.0f,  165.0f,  114.0f, 0.0f },
    {  290.0f,    0.0f,  114.0f, 0.0f },

    {   82.0f,    0.0f,  225.0f, 0.0f },
    {   82.0f,  165.0f,  225.0f, 0.0f },
    {  130.0f,  165.0f,   65.0f, 0.0f },

    {   82.0f,    0.0f,  225.0f, 0.0f },
    {  130.0f,  165.0f,   65.0f, 0.0f },
    {  130.0f,    0.0f,   65.0f, 0.0f },

    {  240.0f,    0.0f,  272.0f, 0.0f },
    {  240.0f,  165.0f,  272.0f, 0.0f },
    {   82.0f,  165.0f,  225.0f, 0.0f },

    {  240.0f,    0.0f,  272.0f, 0.0f },
    {   82.0f,  165.0f,  225.0f, 0.0f },
    {   82.0f,    0.0f,  225.0f, 0.0f },

    // Tall block -- white lambert
    {  423.0f,  330.0f,  247.0f, 0.0f },
    {  265.0f,  330.0f,  296.0f, 0.0f },
    {  314.0f,  330.0f,  455.0f, 0.0f },

    {  423.0f,  330.0f,  247.0f, 0.0f },
    {  314.0f,  330.0f,  455.0f, 0.0f },
    {  472.0f,  330.0f,  406.0f, 0.0f },

    {  423.0f,    0.0f,  247.0f, 0.0f },
    {  423.0f,  330.0f,  247.0f, 0.0f },
    {  472.0f,  330.0f,  406.0f, 0.0f },

    {  423.0f,    0.0f,  247.0f, 0.0f },
    {  472.0f,  330.0f,  406.0f, 0.0f },
    {  472.0f,    0.0f,  406.0f, 0.0f },

    {  472.0f,    0.0f,  406.0f, 0.0f },
    {  472.0f,  330.0f,  406.0f, 0.0f },
    {  314.0f,  330.0f,  456.0f, 0.0f },

    {  472.0f,    0.0f,  406.0f, 0.0f },
    {  314.0f,  330.0f,  456.0f, 0.0f },
    {  314.0f,    0.0f,  456.0f, 0.0f },

    {  314.0f,    0.0f,  456.0f, 0.0f },
    {  314.0f,  330.0f,  456.0f, 0.0f },
    {  265.0f,  330.0f,  296.0f, 0.0f },

    {  314.0f,    0.0f,  456.0f, 0.0f },
    {  265.0f,  330.0f,  296.0f, 0.0f },
    {  265.0f,    0.0f,  296.0f, 0.0f },

    {  265.0f,    0.0f,  296.0f, 0.0f },
    {  265.0f,  330.0f,  296.0f, 0.0f },
    {  423.0f,  330.0f,  247.0f, 0.0f },

    {  265.0f,    0.0f,  296.0f, 0.0f },
    {  423.0f,  330.0f,  247.0f, 0.0f },
    {  423.0f,    0.0f,  247.0f, 0.0f },

    // Ceiling light -- emmissive
    {  343.0f,  548.6f,  227.0f, 0.0f },
    {  213.0f,  548.6f,  227.0f, 0.0f },
    {  213.0f,  548.6f,  332.0f, 0.0f },

    {  343.0f,  548.6f,  227.0f, 0.0f },
    {  213.0f,  548.6f,  332.0f, 0.0f },
    {  343.0f,  548.6f,  332.0f, 0.0f }};

    std::vector<Vec3f> verts_triangle = {
        {-0.5f, -0.5f, 0.f},
        { 0.5f, -0.5f, 0.f},
        { 0.f,   0.5f, 0.f}
    };

    /*
    auto mesh_ptr = std::make_shared<TriangleMesh>(base::Mat4f::identity(), vs,
        std::vector<Vec3f>(), std::vector<Vec2f>(), idx, "test", "shader");
    mesh_ptr->convert_to_4f_alignment();
    */
    auto mesh_ptr = std::make_shared<TriangleArray>(base::Mat4f::identity(), std::move(verts_triangle),
        "test", "test");
    OptixAccel accel(ctx);
    accel.add_trianglearray(mesh_ptr);
    auto root_instance_list = std::vector<std::string> {
        "test"
    };
    //accel.build(root_instance_list);
    auto gas_handle = accel.handles["test"].second;

    CUstream stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    /*
    float3 eye = make_float3(0.f, 5.f, 10.f);
    float3 lookat = make_float3(0.f, 0.f, 0.f);
    */
    //float3 eye = make_float3(278.f, 273.f, -900.f);
    //float3 lookat = make_float3(278.f, 273.f, 330.f);
    float3 eye = make_float3(0.f, 0.f, 2.f);
    float3 lookat = make_float3(0.f, 0.f, 0.f);
    float3 front = normalize(lookat - eye);
    float3 right = cross(front, make_float3(0, 1, 0));
    float3 up = cross(right, front);
    float ratio = static_cast<float>(w) / h;
    float fov = to_radian(45.f / 2.f);
    float scaled_height = std::tan(fov);
    /*
    Params params {
        .image = output,
        .width = static_cast<int>(w),
        .height = static_cast<int>(h),
        .sample_cnt = 5,
        .eye = eye,
        .U = up * scaled_height,
        .V = right * ratio * scaled_height,
        .W = front,
        .handle = accel.get_root_handle()
    };
    */

    ParamsTriangle params {
        .image = output,
        .width = w,
        .height = h,
        .eye = eye,
        .U = right * ratio * scaled_height,
        .V = up * scaled_height,
        .W = front,
        //.handle = accel.get_root_handle()
        .handle = gas_handle
    };

    CUdeviceptr param_ptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&param_ptr),
        sizeof(Params)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(param_ptr), &params,
        sizeof(Params), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(ppl, stream, param_ptr, sizeof(Params),
        &sbt, w, h, 1));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(param_ptr)));

    // Output the image
    std::vector<float> host_data(w * h * 4);
    CUDA_CHECK(cudaMemcpy(host_data.data(), reinterpret_cast<void*>(output),
        buf_size, cudaMemcpyDeviceToHost));
    auto buf_spec = OIIO::ImageSpec(w, h, 4, OIIO::TypeDesc::FLOAT);
    auto image_buf = OIIO::ImageBuf(buf_spec, host_data.data());
    image_buf = OIIO::ImageBufAlgo::flip(image_buf);
    image_buf.write("test.png", OIIO::TypeDesc::UINT8);

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