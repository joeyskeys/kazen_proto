#pragma once

#include <iostream>
#include <iomanip>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)
    {                                                                               \
        cudaError_t res = call;                                                     \
        if (res != cudaSuccess) {                                                   \
            std::cerr << fmt::format("[CUDA ERROR] Cuda call '{}' failed with"      \
                " error: {} ({}:{})").format(#call,                                 \
                cudaGetErrorString(res), __FILE__,  __LINE__) << std::endl;         \
            exit(1);                                                                \
        }                                                                           \
    }

#define OPTIX_CHECK(call)                                                           \
    {                                                                               \
        OptixResult res = call;                                                     \
        if (res != OPTIX_SUCCESS) {                                                 \
            std::cerr << fmt::format("[OPTIX ERROR] OptiX call '{}' failed with"    \
                " error: {} ({}:{})").format(#call, optixGetErrorName(res),         \
                __FILE__, __LINE__) << std::endl;                                   \
            exit(1);                                                                \
        }                                                                           \
    }

#define OPTIX_CHECK_MSG(call, msg)                                                  \
    {                                                                               \
        OptixResult res = call;                                                     \
        if (res != OPTIX_SUCCESS) {                                                 \
            std::cerr << fmt::format("[OPTIX ERROR] OptiX call '{}' failed with"    \
                "error: {} ({}:{})\nMessage: {}").format(#call,                     \
                optixGetErrorName(res), __FILE__, __LINE__, msg);                   \
            exit(1);                                                                \
        }                                                                           \
    }

static void context_log_cb(uint32_t lv, const char* tag, const char* msg, void*) {
    std::cerr << "[" << std::setw(2) << lv << "][" << std::setw(12) << tag << "]: "
        << msg << std::endl;
}

constexpr static uint32_t optix_log_buf_size = 4096;
static char optix_log_buf[optix_log_buf_size];

enum OPTIX_STAGE {
    RAYGEN,
    MISS,
    ANYHIT,
    CLOESTHIT,
    EXCEPTION,
    CALLABLE
};

OptixDeviceContext  create_optix_ctx(const OptixDeviceContextOptions*);
void                destroy_optix_ctx(const OptixDeviceContext);

bool                load_optix_module(const char*, const OptixDeviceContext,
    const OptixModuleCompileOptions*, const OptixPipelineCompileOptions*,
    OptixModule*);
bool                create_optix_pg(const OptixDeviceContext, const OptixProgramGroupDesc*,
    const int, OptixProgramGroupOptions*, OptixProgramGroup*);
bool                link_optix_ppl(const OptixDeviceContext, const OptixPipelineLinkOptions&,
    const OptixPipelineCompileOptions&, const std::vector<OptixProgramGroup>&, OptixPipeline*);
std::vector<GenericRecord> generate_records(const uint32_t group_size);