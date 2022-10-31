#pragma once

#include <limits>

#include <OSL/oslexec.h>

#include "base/casts.h"
#include "base/fp.h"
#include "base/vec.h"
#include "base/utils.h"
#include "core/ray.h"

using base::Vec2f;
using base::Vec3f;
using base::Vec4f;
using base::Vec3i;

class Material;
using MaterialPtr = Material*;

class Shape;
using ShapePtr = Shape*;

struct Intersection {
    Vec3f P;
    Vec3f dpdu;
    Vec3f dpdv;
    Vec3f dpdx;
    Vec3f dpdy;
    Vec3f N;
    Vec3f shading_normal;
    Vec3f tangent;
    Vec3f bitangent;
    Frame frame;
    Vec3f wo;
    Vec3f wi;
    Vec3f bary;
    Vec2f uv;
    Ray*  ray;
    float ray_t = std::numeric_limits<float>::max();
    bool  backface;
    bool  is_light;
    uint  light_id;
    ShapePtr shape;
    size_t geom_id;
    size_t prim_id;
    std::string shader_name;
    OSL::ShaderGroupRef shader;

    Vec3f adaptive_offset_point(int64_t initial_mag);
    Vec3f offset_point1() const;
    float offset_point2() const;

    inline Vec3f to_local(const Vec3f& v) const {
        return frame.to_local(v);
    }

    inline Vec3f to_world(const Vec3f v) const {
        return frame.to_world(v);
    }

    inline void calculate_differentials() {
        float tx, ty;
        if (!plane_intersect(*ray, P, N, tx) || !plane_intersect(*ray, P, N, ty))
            return;
        
        dpdx = base::normalize(ray->origin_dx + tx * ray->direction_dx - P);
        dpdy = base::normalize(ray->origin_dy + ty * ray->direction_dy - P);
    }
};

inline Vec3f Intersection::adaptive_offset_point(int64_t inital_mag) {
    // Checkout appleseed/renderer/kernel/intersection/refining.h
    // And also http://vts.uni-ulm.de/docs/2008/6265/vts_6265_8393.pdf
    return Vec3f{0};
}

inline constexpr float off_origin = 1.f / 32.f;
inline constexpr float off_float_scale = 1.f / 65536.f;
inline constexpr float off_in_scale = 256.f;

// Method from https://link.springer.com/content/pdf/10.1007%2F978-1-4842-4427-2_6.pdf
inline Vec3f Intersection::offset_point1() const {
    Vec3i off_i{off_in_scale * P.x(), off_in_scale * P.y(), off_in_scale * P.z()};
    Vec3f p_i{binary_cast<float>(binary_cast<int>(P.x()) + (P.x() < 0) ? -off_i.x() : off_i.x()),
        binary_cast<float>(binary_cast<int>(P.y()) + (P.y() < 0) ? -off_i.y() : off_i.y()),
        binary_cast<float>(binary_cast<int>(P.z()) + (P.z() < 0) ? -off_i.z() : off_i.z())};
    return Vec3f{fabsf(P.x()) < off_origin ? P.x() + off_float_scale * N.x() : p_i.x(),
        fabsf(P.y()) < off_origin ? P.y() + off_float_scale * N.y() : p_i.y(),
        fabsf(P.z()) < off_origin ? P.z() + off_float_scale * N.z() : p_i.z()};
}

// Method from appleseed, works but cannot totally avoid self-intersection
inline float Intersection::offset_point2() const {
    auto max_dir_component = base::max_component(wi);
    uint32_t max_origin_exp = std::max(std::max(
        FP<float>::exponent(P.x()),
        FP<float>::exponent(P.y())),
        FP<float>::exponent(P.z()));
    
    // Calculate exponent-adaptive offset.
    // Note: float is represented in memory
    // as 1 sign bit, 8 exponent bits and 23 mantissa bits.
    // Higher 24th bit is always 1 in normalized form, hence it's ommited.
    // Mantissa of constructed float will overlap no more than 11 last bits of
    // origin components due to exponent shift.
    // Mantissa of constructed float is just a
    // sequence of 11 ones followed by zeroes.

    const float offset = FP<float>::construct(
        0,
        std::max(static_cast<std::int32_t>(max_origin_exp - 23 + 11), 0),
        2047UL << (23 - 11));

    // Divide by max_dir_component to compensate inverse operation
    // during intersection search. (Actual start point is org + dir * tnear)
    return offset / max_dir_component;
}
