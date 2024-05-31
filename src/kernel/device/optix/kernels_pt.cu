#include <optix.h>
#include <cuda/helpers.h>
#include <cuda/random.h>

#include "types.h"
#include "mathutils.h"

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
    
}
