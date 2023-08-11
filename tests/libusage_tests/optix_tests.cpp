#include <iostream>
#include <iomanip>

#include <catch2/catch_all.hpp>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

#include "base/utils.h"

static void context_log_cb(uint32_t level, const char* tag, const char* message, void*) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
        << message << "\n";
}

TEST_CASE("OptiX initialize", "optix") {
    // Initialize CUDA
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
}