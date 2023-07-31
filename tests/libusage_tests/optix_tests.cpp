#include <catch2/catch_all.hpp>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

TEST_CASE("OptiX initialize", "optix") {
    // Initialize CUDA
    cudaFree(0);

    int num_devices;
    cudaGetDeviceCount(&num_devices);

    if (num_devices == 0)
        throw std::runtime_error("No available CUDA device");

    auto ret = optixInit();
    REQUIRE(ret == OPTIX_SUCCESS);
}