#include <optix.h>
#include <cuda/helpers.h>
#include <cuda/random.h>

#include "kernel/types.h"
#include "kernel/mathutils.h"

extern "C" {
    __constant__ ParamsTriangle params;
}

static __forceinline__ __device__ void set_payload_triangle(float3 p) {
    optixSetPayload_0(__float_as_uint(p.x));
    optixSetPayload_1(__float_as_uint(p.y));
    optixSetPayload_2(__float_as_uint(p.z));
}

extern "C" __global__ void __raygen__rg_triangle() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    float3 ray_origin, ray_direction;
    ray_origin = params.eye;
    const float2 d = 2.f * make_float2(
        static_cast<float>(idx.x) / static_cast<float>(dim.x),
        static_cast<float>(idx.y) / static_cast<float>(dim.y)
    ) - 1.f;
    ray_direction = normalize(d.x * params.U + d.y * params.V + params.W);
    
    unsigned int p0, p1, p2;
    optixTrace(
        params.handle,
        ray_origin,
        ray_direction,
        0.f,
        1e16f,
        0.f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,
        1,
        0,
        p0, p1, p2);
    float3 result;
    result.x = __uint_as_float(p0);
    result.y = __uint_as_float(p1);
    result.z = __uint_as_float(p2);

    auto output = reinterpret_cast<float4*>(params.image);
    output[idx.y * params.width + idx.x] = make_float4(result.x, result.y, result.z, 1.f);
}

extern "C" __global__ void __miss__ms_triangle() {
    MissDataTriangle* miss_data = reinterpret_cast<MissDataTriangle*>(optixGetSbtDataPointer());
    set_payload_triangle(miss_data->bg_color);
}

extern "C" __global__ void __closesthit__ch_triangle() {
    const float2 barycentrics = optixGetTriangleBarycentrics();
    //set_payload_triangle(make_float3(barycentrics, 1.0f));
    set_payload_triangle(make_float3(0.5, 0.7, 0.9));
}