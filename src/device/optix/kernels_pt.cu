#include <optix.h>
#include <cuda/helpers.h>
#include <cuda/random.h>

#include "kernel/types.h"
#include "kernel/mathutils.h"

/***********************
 * We're not going to do wavefront path tracing in GPU right now, stick to
 * the mega kernel way for now, which is not good.
 ***********************/

extern "C" {
    __device__ __constant__ Params params;
}

// Payload pattern from cycles
// 1. ray t
// 2. hit point u
// 3. hit point v
// 4. hit prim number
// 5. hit object idx
// 6. hit object type
static __forceinline__ __device__ void store_rayt(float t) {
    optixSetPayload_0(__float_as_uint(t));
}

static __forceinline__ __device__ void store_u(float u) {
    optixSetPayload_1(__float_as_uint(u));
}

static __forceinline__ __device__ void store_v(float v) {
    optixSetPayload_2(__float_as_uint(v));
}

static __forceinline__ __device__ void store_prim(uint p) {
    optixSetPayload_3(p);
}

static __forceinline__ __device__ void store_obj(uint o) {
    optixSetPayload_4(o);
}

static __forceinline__ __device__ void store_type(uint t) {
    optixSetPayload_5(t);
}

extern "C"
__global__ void __miss__pt() {
    store_rayt(optixGetRayTmax());
    store_type(PRIMITIVE_NONE);
}

extern "C"
__global__ void __raygen__pt() {
    const int    w   = params.width;
    const int    h   = params.height;
    const float3 eye = params.eye;
    const float3 U   = params.U;
    const float3 V   = params.V;
    const float3 W   = params.W;
    const uint3  idx = optixGetLaunchIndex();

    unsigned int seed = tea<4>(idx.y * w + idx.x, 5);

    float3 radiance = make_float3(0.0f);
    int i = params.sample_cnt;
    do {
        // no jitter for now
        const float2 d = 2.f * make_float2(
            static_cast<float>(idx.x) / static_cast<float>(w),
            static_cast<float>(idx.y) / static_cast<float>(h)
            ) - 1.f;
        float3 ray_dir = normalize(d.x * U + d.y * V + W);
        float3 ray_pos = eye;

        ShaderGlobalTmp sg {
            .attenuation    = make_float3(1.f),
            .seed           = seed,
            .depth          = 0
        };

        radiance += integrator_shade_surface(params.handle, ray_pos, ray_dir, sg);
    } while(--i);

    radiance /= params.sample_cnt;
    auto output = reinterpret_cast<float4*>(params.image);
    output[idx.y * w + idx.x] = make_float4(radiance.x, radiance.y, radiance.z, 1.0f);
}
