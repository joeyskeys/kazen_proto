#include <optix.h>
#include <cuda/helpers.h>
#include <cuda/random.h>

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

static __forceinline__ __device__ float3 integrator_li(
    OptixTraversableHandle  handle,
    float3                  ray_pos,
    float3                  ray_dir,
    ShaderGlobalTmp         sg,
    float                   tmin = 0.00001f,
    float                   tmax = 1e16f)
{
    float3 ret = make_float3(0.f);
    while (true) {
        // SER opted ray tracing
        unsigned int u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11,
            u12, u13, u14, u15, u16, u17;

        u0 = __float_as_uint(sg.attenuation.x);
        u1 = __float_as_uint(sg.attenuation.y);
        u2 = __float_as_uint(sg.attenuation.z);
        u3 = sg.seed;
        u4 = sg.depth;

        optixTraverse(
            PAYLOAD_TYPE_RADIANCE,
            handle,
            ray_pos,
            ray_dir,
            tmin,
            tmax,
            0.f, //rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_NONE,
            0,              // SBT offset
            RAY_TYPE_COUNT, //SBT stride
            0,              // missSBTIndex
            u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17
        );

        optixReorder(
            // Check for documentation for proper usage of shader execution reordering
        );

        optixInvoke(PAYLOAD_TYPE_RADIANCE,
            u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17);

        sg.attenuation = make_float3(__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2));
        sg.seed = u3;
        sg.depth = u4;

        sg.emitted = make_float3(__uint_as_float(u5), __uint_as_float(u6), __uint_as_float(u7));
        sg.radiance = make_float3(__uint_as_float(u8), __uint_as_float(u9), __uint_as_float(u10));
        sg.origin = make_float3(__uint_as_float(u11), __uint_as_float(u12), __uint_as_float(u13));
        sg.direction = make_float3(__uint_as_float(u14), __uint_as_float(u15), __uint_as_float(u16));
        sg.done = u17;

        // Trace result computation
        ret += sg.emitted;
        ret += sg.radiance * sg.attenuation;

        const float p = dot( sg.attenuation, make_float3( 0.30f, 0.59f, 0.11f ) );
        const bool done = sg.done  || rnd( sg.seed ) > p;
        if( done )
            break;
        sg.attenuation /= p;

        ray_pos     = sg.origin;
        ray_dir     = sg.direction;

        ++sg.depth;
    }
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

    unsigned int seed = tea<4>(idx.y * w + idx.x, 5);

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

        ShaderGlobalTmp sg {
            .attenuation    = make_float3(1.f),
            .seed           = seed,
            .depth          = 0
        };

        radiance += integrator_li(params.handle, ray_pos, ray_dir, sg);
    } while(--i);

    radiance /= sample_cnt;
    auto output = reinterpret_cast<float3*>(params.image);
    output[idx.y * w + idx.x] = radiance;
}

static __forceinline__ __device__ void store_miss_radiace_sg(ShaderGlobalTmp sg) {
    optixSetPayload_5(__float_as_uint(sg.emitted.x));
    optixSetPayload_6(__float_as_uint(sg.emitted.y));
    optixSetPayload_7(__float_as_uint(sg.emitted.z));

    optixSetPayload_8(__float_as_uint(sg.radiance.x));
    optixSetPayload_9(__float_as_uint(sg.radiance.y));
    optixSetPayload_10(__float_as_uint(sg.radiance.z));

    optixSetPayload_17(sg.done);
}

extern "C" __global__ void __miss__radiance() {
    optixSetPayloadTypes(PAYLOAD_TYPE_RADIANCE);
    auto data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    ShaderGlobalTmp sg{};

    sg.radiance = make_float3(data->bg_color);
    sg.emitted  = make_float3(0.f);
    sg.done     = true;

    store_miss_radiace_sg(sg);
}

extern "C" __global__ void __closesthit_radiance() {

}