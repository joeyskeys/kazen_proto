#pragma once

#include <iostream>
#include <iomanip>

#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "kernel/types.h"
#include "kernel/oslutils.h"

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

#define NVRTC_CHECK( call )                                                         \
    {                                                                               \
        nvrtcResult code = call;                                                    \
        if (code != NVRTC_SUCCESS) {                                                \
            fmt::print(stderr, "[NVRTC ERROR] NVRTC call '{}' failed with "         \
                "error: {} ({}:{})", #call,                                         \
                nvrtcGetErrorString(code), __FILE__, __LINE__);                     \
            exit(1);                                                                \
        }                                                                           \
    }

static void context_log_cb(uint32_t lv, const char* tag, const char* msg, void*) {
    std::cerr << "[" << std::setw(2) << lv << "][" << std::setw(12) << tag << "]: "
        << msg << std::endl;
}

#define LOG_BUF_SIZE 4096
static char log_buf[LOG_BUF_SIZE];
static size_t log_size = 0;

enum OPTIX_STAGE {
    RAYGEN,
    MISS,
    CLOSESTHIT,
    ANYHIT,
    EXCEPTION,
    CALLABLE
};

#define CUDA_NVRTC_OPTIONS \
    "-std=c++20", \
    "-arch", \
    "compute_50", \
    "-use_fast_math", \
    "-lineinfo", \
    "-default-device", \
    "-rdc", \
    "true", \
    "-D__x86_64", \
    "-DOPTIX_OPTIONAL_FEATURE_OPTIX7", \
    "--device-debug"

// TODO : remove it later
#define OPTIX_INC_DIRS \
    "-I/home/joey/Desktop/softs/optix/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/include", \
    "-I/home/joey/Desktop/softs/optix/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK", \
    "-I/opt/cuda/include", \
    "-I/home/joey/Desktop/repos/kazen_proto/src/kernel",

OptixDeviceContext  create_optix_ctx(const OptixDeviceContextOptions*);
void                destroy_optix_ctx(const OptixDeviceContext);

std::string         cu_to_ptx(const char*, const char*,
    const std::vector<const char*>& inc_dirs = {OPTIX_INC_DIRS},
    const std::vector<const char*>& compiler_options = {CUDA_NVRTC_OPTIONS});
bool                load_optix_module_ptx(const char*, const OptixDeviceContext,
    const OptixModuleCompileOptions*, const OptixPipelineCompileOptions*,
    OptixModule*);
bool                load_optix_module_cu(const char*, const OptixDeviceContext,
    const OptixModuleCompileOptions*, const OptixPipelineCompileOptions*,
    OptixModule*);
bool                load_raw_ptx(const char*, const OptixDeviceContext,
    const OptixModuleCompileOptions*, const OptixPipleineCompileOptions*,
    OptixModule*);
bool                create_optix_pg(const OptixDeviceContext, const OptixProgramGroupDesc*,
    const int, OptixProgramGroupOptions*, OptixProgramGroup*);
bool                create_optix_ppl(const OptixDeviceContext, const OptixPipelineCompileOptions&,
    const OptixPipelineLinkOptions&, const std::vector<OptixProgramGroup>&, OptixPipeline*);
std::vector<GenericRecord> generate_records(const uint32_t group_size);

std::pair<std::vector<OptixProgramGroup>, std::vector<void*>> create_osl_pgs(
    const OptixDeviceContext, const OptixModuleCompileOptions&, const OptixPipelineCompileOptions&,
    const PTXMap&);