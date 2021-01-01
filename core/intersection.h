#pragma once

#include "base/vec.h"

struct Intersection {
    Vec3f p;
    Vec3f n;
    Vec3f ng;
    Vec3f t;
    Vec3f b;
    Vec3f bary;
    Vec2f uv;
};