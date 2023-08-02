#include <iostream>
#include <iomanip>

#include <catch2/catch_all.hpp>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

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
}