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

struct ParamsForTest {
    uchar4* image;
    uint32_t image_width;
};