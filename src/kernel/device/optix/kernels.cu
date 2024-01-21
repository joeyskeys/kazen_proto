#include <optix.h>
#include <cuda/helpers.h>

#include "types.h"

extern "C" {
    __constant__ ParamsForTest params;
}

extern "C"
__global__ void __raygen__fixed() {
    uint3 launch_idx = optixGetLaunchIndex();
    Pixel* data = reinterpret_cast<Pixel*>(optixGetSbtDataPointer());
    params.pixels[launch_idx.y * params.image_width + launch_idx.x] =
        make_color(make_float3(data->r, data->g, data->b));
}