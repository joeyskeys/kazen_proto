
#include <optix.h>
#include <cuda/helpers.h>

#include "solid.h"

extern "C" {
    __constant__ Params params;
}

extern "C"
__global__ void __raygen__draw_solid_color() {
    uint3 launch_index = optixGetLaunchIndex();
    RayGenData* rt_data = (RayGenData*)optixGetSbtDataPointer();
    params.image[launch_index.y * params.image_width + launch_index.x] =
        make_color(make_float3(rt_data->r, rt_data->g, rt_data->b));
}