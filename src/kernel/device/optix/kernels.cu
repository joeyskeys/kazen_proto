#include <optix.h>
#include <cuda/helpers.h>
#include <cuda/random.h>

#include "types.h"
#include "vec_math.h"

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

static __forceinline__ __device__ void store_MS_sg(ShaderGlobalTmp sg) {
    optixSetPayload_5(__float_as_uint(sg.emitted.x));
    optixSetPayload_6(__float_as_uint(sg.emitted.y));
    optixSetPayload_7(__float_as_uint(sg.emitted.z));

    optixSetPayload_8(__float_as_uint(sg.radiance.x));
    optixSetPayload_9(__float_as_uint(sg.radiance.y));
    optixSetPayload_10(__float_as_uint(sg.radiance.z));

    optixSetPayload_17(sg.done);
}

static __forceinline__ __device__ ShaderGlobalTmp load_CH_sg() {
    ShaderGlobalTmp sg{};
    sg.attenuation.x = __uint_as_float(optixGetPayload_0());
    sg.attenuation.y = __uint_as_float(optixGetPayload_1());
    sg.attenuation.z = __uint_as_float(optixGetPayload_2());
    sg.seed = optixGetPayload_3();
    sg.depth = optixGetPayload_4();
    return sg;
}

static __forceinline__ __device__ void store_CH_sg(ShaderGlobalTmp sg) {
    optixSetPayload_0( __float_as_uint( sg.attenuation.x ) );
    optixSetPayload_1( __float_as_uint( sg.attenuation.y ) );
    optixSetPayload_2( __float_as_uint( sg.attenuation.z ) );

    optixSetPayload_3( sg.seed );
    optixSetPayload_4( sg.depth );

    optixSetPayload_5( __float_as_uint( sg.emitted.x ) );
    optixSetPayload_6( __float_as_uint( sg.emitted.y ) );
    optixSetPayload_7( __float_as_uint( sg.emitted.z ) );

    optixSetPayload_8( __float_as_uint( sg.radiance.x ) );
    optixSetPayload_9( __float_as_uint( sg.radiance.y ) );
    optixSetPayload_10( __float_as_uint( sg.radiance.z ) );

    optixSetPayload_11( __float_as_uint( sg.origin.x ) );
    optixSetPayload_12( __float_as_uint( sg.origin.y ) );
    optixSetPayload_13( __float_as_uint( sg.origin.z ) );

    optixSetPayload_14( __float_as_uint( sg.direction.x ) );
    optixSetPayload_15( __float_as_uint( sg.direction.y ) );
    optixSetPayload_16( __float_as_uint( sg.direction.z ) );

    optixSetPayload_17( sg.done );
}

extern "C" __global__ void __miss__radiance() {
    optixSetPayloadTypes(PAYLOAD_TYPE_RADIANCE);
    auto data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    ShaderGlobalTmp sg{};

    sg.radiance = make_float3(data->bg_color);
    sg.emitted  = make_float3(0.f);
    sg.done     = true;

    store_MS_sg(sg);
}

struct Onb
{
  __forceinline__ __device__ Onb(const float3& normal)
  {
    m_normal = normal;

    if( fabs(m_normal.x) > fabs(m_normal.z) )
    {
      m_binormal.x = -m_normal.y;
      m_binormal.y =  m_normal.x;
      m_binormal.z =  0;
    }
    else
    {
      m_binormal.x =  0;
      m_binormal.y = -m_normal.z;
      m_binormal.z =  m_normal.y;
    }

    m_binormal = normalize(m_binormal);
    m_tangent = cross( m_binormal, m_normal );
  }

  __forceinline__ __device__ void inverse_transform(float3& p) const
  {
    p = p.x*m_tangent + p.y*m_binormal + p.z*m_normal;
  }

  float3 m_tangent;
  float3 m_binormal;
  float3 m_normal;
};

static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
  // Uniformly sample disk.
  const float r   = sqrtf( u1 );
  const float phi = 2.0f*M_PIf * u2;
  p.x = r * cosf( phi );
  p.y = r * sinf( phi );

  // Project up to hemisphere.
  p.z = sqrtf( fmaxf( 0.0f, 1.0f - p.x*p.x - p.y*p.y ) );
}

static __forceinline__ __device__ bool trace_occlusion(
    OptixTraversableHandle  handle,
    float3                  ray_pos,
    float3                  ray_dir,
    float                   tmin,
    float                   tmax)
{
    optixTraverse(
        handle,
        ray_pos,
        ray_dir,
        tmin,
        tmax,
        0.f, // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0, //SBT offset
        RAY_TYPE_COUNT, // SBT stride
        0  // miss SBT index
    );
    return optixHitObjectIsHit();
}

extern "C" __global__ void __closesthit_radiance() {
    optixSetPayloadTypes(PAYLOAD_TYPE_RADIANCE);
    HitGroupData* rt_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());

    const int prim_idx = optixGetPrimitiveIndex();
    const float3 ray_dir = optixGetWorldRayDirection();
    const int vert_idx_offset = prim_idx * 3;

    const float3 v0 = make_float3(rt_data->vertices[vert_idx_offset    ]);
    const float3 v1 = make_float3(rt_data->vertices[vert_idx_offset + 1]);
    const float3 v2 = make_float3(rt_data->vertices[vert_idx_offset + 2]);
    const float3 N_0 = normalize(cross(v1 - v0, v2 - v0));
    const float3 N  = faceforward(N_0, -ray_dir, N_0);
    const float3 P  = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;
    ShaderGlobalTmp sg = load_CH_sg();

    if (sg.depth == 0)
        sg.emitted = rt_data->emission_color;
    else
        sg.emitted = make_float3(0.f);

    unsigned int seed = sg.seed;
    {
        const float z1 = rnd(seed);
        const float z2 = rnd(seed);
        float3 w_in;
        cosine_sample_hemisphere(z1, z2, w_in);
        Onb onb(N);
        onb.inverse_transform(w_in);
        sg.direction = w_in;
        sg.origin = P;
        sg.attenuation *= rt_data->diffuse_color;
    }

    // fake a light for now
    const float3 light_pos = make_float3(0.f, 10.f, 0.f);
    const float  Ldist = length(light_pos - P);
    const float3 L     = normalize(light_pos - P);
    const float  nDl   = dot(N, L);
    const float  LnDl  = -dot(make_float3(0.f, -1.f, 0.f), L);

    float weight = 0.f;
    if (nDl > 0.f && LnDl > 0.f) {
        const bool occluded = trace_occlusion(
            params.handle,
            P,
            L,
            0.0001f,
            Ldist - 0.0001f);

        if (!occluded) {
            const float A = 5.f; // fake area data
            weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
        }
    }

    sg.radiance = light.emission * weight;
    sg.done = false;

    store_CH_sg(sg);
}