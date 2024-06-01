#pragma once

#include "kernel/types.h"

static __forceinline__ __device__ float3 integrator_shade_surface(
    OptixTraversableHandle  handle,
    float3                  ray_pos,
    float3                  ray_dir,
    ShaderGlobal            sg,
    float                   tmin = 1e-5f,
    float                   tmax = 1e16f)
{
    
}