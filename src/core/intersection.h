#pragma once

#include <limits>

#include <OSL/oslexec.h>

#include "base/vec.h"
#include "base/utils.h"

class Material;
using MaterialPtr = Material*;

class Shape;
using ShapePtr = Shape*;

struct Intersection {
    Vec3f position;
    Vec3f refined_point;
    Vec3f normal;
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

    void adaptive_offset_point(int64_t initial_mag);
};

inline void Intersection::adaptive_offset_point(int64_t inital_mag) {
    // Checkout appleseed/renderer/kernel/intersection/refining.h
    // And also http://vts.uni-ulm.de/docs/2008/6265/vts_6265_8393.pdf
}