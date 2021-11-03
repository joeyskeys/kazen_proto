#pragma once

#include "base/vec.h"
#include "base/utils.h"

class Material;
using MaterialPtr = Material*;

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
    float ray_t;
    bool  backface;
    MaterialPtr mat;
    uint  shader_id;
    uint  obj_id;
};