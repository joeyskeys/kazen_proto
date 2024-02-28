#pragma once

#include <vector_types.h>

#include "base/vec.h"

// Host side conversion func
template <typename T>
auto convert_to_cuda_type(const T& a) {
    if constexpr (sizeof(T) == 8)
        return make_float2(a[0], a[1]);
    else if constexpr (sizeof(T) == 12)
        return make_float3(a[0], a[1], a[2]);
    else
        return make_float4(a[0], a[1], a[2], a[3]);
}

inline float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
