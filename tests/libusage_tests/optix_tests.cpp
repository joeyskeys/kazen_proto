#include <iostream>
#include <iomanip>

#include <catch2/catch_all.hpp>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

#include "base/utils.h"
#include "solid.h"

static void context_log_cb(uint32_t level, const char* tag, const char* message, void*) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
        << message << "\n";
}

template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RayGenSbtRecord = SbtRecord<RayGenData>;
using MissSbtRecord = SbtRecord<int>;

TEST_CASE("OptiX initialize", "optix") {
    // Initialize CUDA and create OptiX context
    cudaFree(0);

    int num_devices;
    cudaGetDeviceCount(&num_devices);

    if (num_devices == 0)
        throw std::runtime_error("No available CUDA device");

    auto ret = optixInit();
    REQUIRE(ret == OPTIX_SUCCESS);

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;

    CUcontext cu_ctx = 0; // Take the current context
    OptixDeviceContext optix_ctx;
    ret = optixDeviceContextCreate(cu_ctx, &options, &optix_ctx);
    REQUIRE(ret == OPTIX_SUCCESS);

    // Create module
    OptixModule module = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options = {};

    OptixModuleCompileOptions module_compile_options = {};
#ifndef NDEBUG
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_compile_options.numPayloadValues = 2;
    pipeline_compile_options.numAttributeValues = 2;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    size_t input_size = 0;
    const std::string cu_content = load_file("./tests/libusage_tests/solid.ptx");
    char log_buf[2048];
    size_t buf_size = sizeof(log_buf);

    ret = optixModuleCreate(
        optix_ctx,
        &module_compile_options,
        &pipeline_compile_options,
        cu_content.c_str(),
        cu_content.size(),
        log_buf, &buf_size,
        &module
    );

    REQUIRE(ret == OPTIX_SUCCESS);

    // Create program groups
    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group   = nullptr;

    OptixProgramGroupOptions    prog_group_options = {};
    OptixProgramGroupDesc       raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__draw_solid_color";
    ret = optixProgramGroupCreate(
        optix_ctx,
        &raygen_prog_group_desc,
        1,
        &prog_group_options,
        log_buf, &buf_size,
        &raygen_prog_group);

    REQUIRE(ret == OPTIX_SUCCESS);

    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    optixProgramGroupCreate(
        optix_ctx,
        &miss_prog_group_desc,
        1,
        &prog_group_options,
        log_buf, &buf_size,
        &miss_prog_group);

    REQUIRE(ret == OPTIX_SUCCESS);

    // Link pipeline
    OptixPipeline ppl = nullptr;
    const uint32_t max_trace_depth = 0;
    OptixProgramGroup prog_groups[] = { raygen_prog_group };
    OptixPipelineLinkOptions ppl_link_options = {};
    ppl_link_options.maxTraceDepth = max_trace_depth;
    ret = optixPipelineCreate(
        optix_ctx,
        &pipeline_compile_options,
        &ppl_link_options,
        prog_groups,
        sizeof(prog_groups) / sizeof(prog_groups[0]),
        log_buf, &buf_size,
        &ppl);

    REQUIRE(ret == OPTIX_SUCCESS);

    OptixStackSizes stack_sizes = {};
    for (auto& prog_group : prog_groups) {
        ret = optixUtilAccumulateStackSizes(prog_group, &stack_sizes, ppl);
        REQUIRE(ret == OPTIX_SUCCESS);
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    ret = optixUtilComputeStackSizes(
        &stack_sizes,
        max_trace_depth,
        0,
        0,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size);

    REQUIRE(ret == OPTIX_SUCCESS);

    ret = optixPipelineSetStackSize(
        ppl,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        2);
    
    REQUIRE(ret == OPTIX_SUCCESS);

    // Setup shader binding table
    OptixShaderBindingTable sbt{};
    // records
    CUdeviceptr raygen_rec;
    const size_t raygen_rec_size = sizeof(RayGenSbtRecord);
    cudaMalloc(reinterpret_cast<void**>(&raygen_rec), raygen_rec_size);
    RayGenSbtRecord rg_sbt;
    ret = optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt);
    REQUIRE(ret == OPTIX_SUCCESS);
    rg_sbt.data = {0.4f, 0.7f, 0.f};
    cudaMemcpy(
        reinterpret_cast<void*>(raygen_rec),
        &rg_sbt,
        raygen_rec_size,
        cudaMemcpyHostToDevice);

    CUdeviceptr miss_rec;
    size_t miss_rec_size = sizeof(MissSbtRecord);
    cudaMalloc(reinterpret_cast<void**>(&miss_rec), miss_rec_size);
    RayGenSbtRecord ms_sbt;
    ret = optixSbtRecordPackHeader(miss_prog_group, &ms_sbt);
    REQUIRE(ret == OPTIX_SUCCESS);
    cudaMemcpy(
        reinterpret_cast<void*>(miss_rec),
        &ms_sbt,
        miss_rec_size,
        cudaMemcpyHostToDevice);

    sbt.raygenRecord = raygen_rec;
    sbt.missRecordBase = miss_rec;
    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt.missRecordCount = 1;

    cudaFree(reinterpret_cast<void*>(sbt.raygenRecord));
    cudaFree(reinterpret_cast<void*>(sbt.missRecordBase));

    ret = optixPipelineDestroy(ppl);
    REQUIRE(ret == OPTIX_SUCCESS);
    ret = optixProgramGroupDestroy(miss_prog_group);
    REQUIRE(ret == OPTIX_SUCCESS);
    ret = optixProgramGroupDestroy(raygen_prog_group);
    REQUIRE(ret == OPTIX_SUCCESS);
    ret = optixModuleDestroy(module);
    REQUIRE(ret == OPTIX_SUCCESS);
    ret = optixDeviceContextDestroy(optix_ctx);
    REQUIRE(ret == OPTIX_SUCCESS);
}