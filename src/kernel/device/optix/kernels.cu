#include <optix.h>
#include <cuda/helpers.h>

#include "types.h"

extern "C" {
    __constant__ Params params;
}

extern "C"
__global__ void __raygen__fixed() {
    uint3 launch_idx = optixGetLaunchIndex();
    Pixel* data = reinterpret_cast<Pixel*>(optixGetSbtDataPointer());
    float3* output = reinterpret_cast<float3*>(params.image);
    output[launch_idx.y * params.width + launch_idx.x] =
        make_float3(data->r, data->g, data->b);
}

extern "C"
__global__ void __raygen__main() {
    const int    w   = params.width;
    const int    h   = params.height;
    const float3 eye = params.eye;
    const float3 U   = params.U;
    const float3 V   = params.V;
    const float3 W   = params.W;
    const uint3  idx = optixGetLaunchIndex();

    float3 radiance = make_float3(0.0f);
    int i = params.sample_cnt;
    do {
        // no jitter for now
        const float2 d = 2.f * make_float2(
            static_cast<float>(idx.x) / static_cast<float>(w),
            static_cast<float>(idx.y) / static_cast<float>(h)
            ) - 1.f;
        float3 ray_dir = normalize(d.x * U + d.y * V + w);
        float3 ray_pos = eye;

        for ( ;; ) {
            break;
        }

    } while(--i);

    radiance /= sample_cnt;
    auto output = reinterpret_cast<float3*>(params.image);
    output[idx.y * w + idx.x] = radiance;
}