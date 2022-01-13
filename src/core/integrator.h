#pragma once

#include <memory>
#include <unordered_map>

#include <OpenImageIO/imageio.h>

#include "camera.h"
#include "film.h"
#include "accel.h"
#include "light.h"
#include "shading/renderservices.h"

class Integrator {
public:
    Integrator();
    Integrator(Camera* cam_ptr, Film* flm_ptr);

    //void render();

    virtual void Li(const Ray& r) const = 0;

    //Sphere*           sphere;
    //ListAccel*        accel_ptr;
    HitablePtr          accel_ptr;
    Camera*             camera_ptr;
    Film*               film_ptr;
    std::vector<std::unique_ptr<Light>>* lights;

    OSL::ShadingSystem*  shadingsys;
    //KazenRenderServices rend;
    std::unordered_map<std::string, OSL::ShaderGroupRef>* shaders;
};

class NormalIntegrator : public Integrator {
    NormalIntegrator();
    NormalIntegrator(Camera* cam_ptr, Film* flm_ptr);

    RGBSpectrum Li(const Ray& r) const override;
};