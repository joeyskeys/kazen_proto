#pragma once

#include <iostream>
#include <iomanip>

#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "kernel/types.h"

#define CUDA_CHECK( call )                                                          \
    {                                                                               \
        cudaError_t res = call;                                                     \
        if (res != cudaSuccess) {                                                   \
            fmt::print(stderr, "[CUDA ERROR] Cuda call '{}' failed with "           \
                "error: {} ({}:{})", #call, cudaGetErrorString(res),                \
                __FILE__,  __LINE__);                                               \
            exit(1);                                                                \
        }                                                                           \
    }

#define OPTIX_CHECK( call )                                                         \
    {                                                                               \
        OptixResult res = call;                                                     \
        if (res != OPTIX_SUCCESS) {                                                 \
            fmt::print(stderr, "[OPTIX ERROR] OptiX call '{}' failed with "         \
                "error: {} ({}:{})", #call, optixGetErrorName(res), __FILE__,       \
                __LINE__);                                                          \
            exit(1);                                                                \
        }                                                                           \
    }

#define OPTIX_CHECK_MSG( call, msg )                                                \
    {                                                                               \
        OptixResult res = call;                                                     \
        if (res != OPTIX_SUCCESS) {                                                 \
            fmt::print(stderr, "[OPTIX ERROR] OptiX call '{}' failed with "         \
                "error: {} ({}:{})\nMessage: {}", #call,                            \
                optixGetErrorName(res), __FILE__, __LINE__, msg);                   \
            exit(1);                                                                \
        }                                                                           \
    }

static void context_log_cb(uint32_t lv, const char* tag, const char* msg, void*) {
    std::cerr << "[" << std::setw(2) << lv << "][" << std::setw(12) << tag << "]: "
        << msg << std::endl;
}

#define OPTIX_LOG_BUF_SIZE 4096
static char optix_log_buf[OPTIX_LOG_BUF_SIZE];
static size_t log_size = 0;

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

void                cu_to_ptx(std::string& output_ptx, const char* inc_dir, const char* cu_src,
    const char* name, const char** log, const std::vector<const char*>& options);
bool                load_optix_module(const char*, const OptixDeviceContext,
    const OptixModuleCompileOptions*, const OptixPipelineCompileOptions*,
    OptixModule*);
bool                create_optix_pg(const OptixDeviceContext, const OptixProgramGroupDesc*,
    const int, OptixProgramGroupOptions*, OptixProgramGroup*);
bool                create_optix_ppl(const OptixDeviceContext, const OptixPipelineCompileOptions&,
    const OptixPipelineLinkOptions&, const std::vector<OptixProgramGroup>&, OptixPipeline*);
std::vector<GenericRecord> generate_records(const uint32_t group_size);