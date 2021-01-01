#pragma once

#include <memory>

#include "camera.h"
#include "film.h"
#include "shape.h"

class Integrator {
public:
    Integrator(Camera* cam_ptr, Film* flm_ptr)
        : camera_ptr(cam_ptr)
        , film_ptr(flm_ptr)
    {}

    void render();

    Sphere*     sphere;

protected:
    Camera*     camera_ptr;
    Film*       film_ptr;
};