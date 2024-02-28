#pragma once

#include <cuda/std/type_traits>

#include <vector_functions.h>
#include <vector_types.h>

#include "device/optix/device_config.h"

// TODO: unify these interface with the CPU side code which will help
// to unify the general kernal code(XPU)

#ifndef M_PIf
#define M_PIf       3.14159265358979323846f
#endif
#ifndef M_PI_2f
#define M_PI_2f     1.57079632679489661923f
#endif
#ifndef M_1_PIf
#define M_1_PIf     0.318309886183790671538f
#endif

#define VEC_OP_TEMPLATE(OP) \
template <typename T> \
kazen_inline kazen_hostdevice T operator OP (const T& a, const T& b) { \
    if constexpr (sizeof(T) == 8) \
        return make_float2(a.x OP b.x, a.y OP b.y); \
    else if constexpr (sizeof(T) == 12) \
        return make_float3(a.x OP b.x, a.y OP b.y, a.z OP b.z); \
    else \
        return make_float4(a.x OP b.x, a.y OP b.y, a.z OP b.z, a.w OP b.w); \
}

#define VEC_OP_ASSIGN_TEMPLATE(OP) \
template <typename T> \
kazen_inline kazen_hostdevice T operator OP##= (T& a, const T& b) { \
    a.x OP##= b.x; a.y OP##= b.y; \
    if constexpr (sizeof(T) > 8) \
        a.z OP##= b.z; \
    if constexpr (sizeof(T) > 12) \
        a.w OP##= b.w; \
}

#define VEC_OP(OP) VEC_OP_TEMPLATE(OP) VEC_OP_ASSIGN_TEMPLATE(OP)

#define VEC_OP_SCALAR_TEMPLATE(OP) \
template <typename T> \
kazen_inline kazen_hostdevice T operator OP (const T& a, const float& b) { \
    if constexpr (sizeof(T) == 8) \
        return make_float2(a.x OP b, a.y OP b); \
    else if constexpr (sizeof(T) == 12) \
        return make_float3(a.x OP b, a.y OP b, a.z OP b); \
    else \
        return make_float4(a.x OP b, a.y OP b, a.z OP b, a.w OP b); \
}

#define VEC_OP_ASSIGN_SCALAR_TEMPLATE(OP) \
template <typename T> \
kazen_inline kazen_hostdevice T operator OP##= (const T& a, const float& b) { \
    a.x OP##= b; a.y OP##= b; \
    if constexpr (sizeof(T) > 8) \
        a.z OP##= b; \
    if constexpr (sizeof(T) > 12) \
        a.w OP##= b; \
}

#define VEC_OP_WITH_SCALAR(OP) \
    VEC_OP(OP) \
    VEC_OP_SCALAR_TEMPLATE(OP) \
    VEC_OP_ASSIGN_SCALAR_TEMPLATE(OP)

VEC_OP(+)
VEC_OP(-)
VEC_OP_WITH_SCALAR(*)
VEC_OP_WITH_SCALAR(/)

template <typename T>
concept cuda_floats = cuda::std::is_same_v<float2, T> &&
    cuda::std::is_same_v<float3, T> && cuda::std::is_same_v<float4, T>;

// Special one for operator *
template <cuda_floats T>
kazen_inline kazen_hostdevice T operator* (const float s, const T& a) {
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

template <typename T>
kazen_inline kazen_hostdevice T lerp(const T& a, const T& b, const float t) {
    return a + (b - a) * t;
}

template <typename T>
kazen_inline kazen_hostdevice T bilerp(const T& x00, const T& x10, const T& x01, const T& x11,
    const float u, const float v)
{
    return lerp(lerp(x00, x10, u), lerp(x01, x11, u), v);
}

template <typename T>
kazen_inline kazen_hostdevice float hsum(const T& a) {
    float sum = 0.f;
    if constexpr (sizeof(T) == 8)
        sum = a.x + a.y;
    else if constexpr (sizeof(T) == 12)
        sum = a.x + a.y + a.z;
    else
        sum = a.x + a.y + a.z + a.w;
    return sum;
}

template <typename T>
kazen_inline kazen_hostdevice float dot(const T& a, const T& b) {
    return hsum(a * b);
}

template <typename T>
kazen_inline kazen_hostdevice float length(const T& a) {
    return sqrtf(dot(a, a));
}

template <typename T>
kazen_inline kazen_hostdevice T normalize(const T& a) {
    // This may have problem since defined for both host&device but
    // the rsqrtf func without any header is only available in device
    // code, same problem for above func
#ifdef __NVCC__
    return a * rsqrtf(dot(a, a));
#else
    auto rsqrt_v = 1.f / sqrtf(dot(a, a));
    return a * rsqrt_v;
#endif
}

template <typename T>
kazen_inline kazen_hostdevice T reflect(const T& i, const T& n) {
    return i - 2.f * n * dot(n, i);
}

template <typename T>
kazen_inline kazen_hostdevice T faceforward(const T& n, const T& i, const T& nref) {
    return n * copysignf(1.f, dot(i, nref));
}
