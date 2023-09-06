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
    CUDeviceptr*    pixels;
    uint32_t        image_width;
};