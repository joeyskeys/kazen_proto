#pragma once

#include "vec.h"
#include "film.h"
#include "ray.h"

class Camera {
public:
    Camera(const Film* f)
        : film(f)
    {}

    Ray generate_ray(uint x, uint y);

private:
    Vec3f position;
    Vec3f lookat;
    Vec3f up;
}