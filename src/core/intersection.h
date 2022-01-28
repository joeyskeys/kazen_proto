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
    ShapePtr shape;
    size_t geom_id;
    std::string shader_name;
    OSL::ShaderGroupRef shader;
};