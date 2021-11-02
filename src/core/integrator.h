#pragma once

#include <memory>

#include <OpenImageIO/imageio.h>

#include "camera.h"
#include "film.h"
#include "accel.h"
#include "light.h"
#include "shading/renderservices.h"

class Integrator {
public:
    Integrator(Camera* cam_ptr, Film* flm_ptr);

    void render();

    //Sphere*           sphere;
    //ListAccel*        accel_ptr;
    HitablePtr          accel_ptr;
    Camera*             camera_ptr;
    Film*               film_ptr;
    std::vector<Light*> lights;

    std::unique_ptr<OSL::ShadingSystem>  shadingsys;
    KazenRenderServices rend;
    OSL::ErrorHandler   errhandler;
};