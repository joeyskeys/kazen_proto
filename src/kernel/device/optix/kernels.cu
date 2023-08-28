#include <optix.h>

#include <cuda/helpers.h>

extern "C"
__global__ void __raygen__fixed() {
    uint3 launch_idx = optixGetLaunchIndex();
    Pixel* data = reinterpret_cast<Pixel*>(optixGetSbtDataPointer());
    params.image[lauch_idx.y * params.image_width + launch_idx.x] =
        make_color(make_float3(data->r, data->g, data->b));
}