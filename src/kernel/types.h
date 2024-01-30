#pragma once

#include <optix.h>

struct Pixel {
    float r;
    float g;
    float b;
};

struct GenericRecord {
    __align__(
        OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void* data;
};

template <typename T>
struct GenericLocalRecord {
    __align__(
        OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct ParamsForTest {
    CUdeviceptr     image;
    unsigned        image_width;
};

struct Params {
    // Output
    CUdeviceptr     image;
    // Film info
    int             width;
    int             height;
    // Sampling info
    int             sample_cnt;
    // Camera info
    float3          eye;
    float3          U;
    float3          V;
    float3          W;
    // Accel
    OptixTraversableHandle handle;
};

struct ShaderGlobalTmp {
    float3          attenuation;
    unsigned int    seed;
    int             depth;

    float3          emitted;
    float3          radiance;
    float3          origin;
    float3          direction;
    int             done;
};

constexpr unsigned int RAY_TYPE_COUNT = 1;
constexpr OptixPayloadTypeID PAYLOAD_TYPE_RADIANCE = OPTIX_PAYLOAD_TYPE_ID_0;

struct MissData {
    float4 bg_color;
};

struct RaygenData {};

struct HitGroupData {
    float3 emission_color;
    float3 diffuse_color;
    float4* vertices;
};