#pragma once

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
}