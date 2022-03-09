#pragma once

#include <limits>

#include <OSL/oslexec.h>

#include "base/casts.h"
#include "base/vec.h"
#include "base/utils.h"

class Material;
using MaterialPtr = Material*;

class Shape;
using ShapePtr = Shape*;

struct Intersection {
    Vec3f P;
    Vec3f refined_point;
    Vec3f N;
    Vec3f shading_normal;
    Vec3f tangent;
    Vec3f bitangent;
    Vec3f wo;
    Vec3f wi;
    Vec3f bary;
    Vec2f uv;
    float ray_t = std::numeric_limits<float>::max();
    bool  backface;
    bool  is_light;
    uint  light_id;
    ShapePtr shape;
    size_t geom_id;
    std::string shader_name;
    OSL::ShaderGroupRef shader;

    Vec3f adaptive_offset_point(int64_t initial_mag);
    Vec3f offset_point1();
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
inline Vec3f Intersection::offset_point1() {
    Vec3i off_i{off_in_scale * P.x(), off_in_scale * P.y(), off_in_scale * P.z()};
    Vec3f p_i{binary_cast<float>(binary_cast<int>(P.x()) + (P.x() < 0) ? -off_i.x() : off_i.x()),
        binary_cast<float>(binary_cast<int>(P.y()) + (P.y() < 0) ? -off_i.y() : off_i.y()),
        binary_cast<float>(binary_cast<int>(P.z()) + (P.z() < 0) ? -off_i.z() : off_i.z())};
    return Vec3f{fabsf(P.x()) < off_origin ? P.x() + off_float_scale * N.x() : p_i.x(),
        fabsf(P.y()) < off_origin ? P.y() + off_float_scale * N.y() : p_i.y(),
        fabsf(P.z()) < off_origin ? P.z() + off_float_scale * N.z() : p_i.z()};
}