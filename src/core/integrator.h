#pragma once

#include <functional>
#include <memory>
#include <unordered_map>

#include <OpenImageIO/imageio.h>

#include "core/accel.h"
#include "core/camera.h"
#include "core/film.h"
#include "core/light.h"
#include "shading/renderservices.h"

class Integrator {
public:
    Integrator();
    Integrator(Camera* cam_ptr, Film* flm_ptr);

    //void render();

    virtual RGBSpectrum Li(const Ray& r) const = 0;

    //Sphere*           sphere;
    //ListAccel*        accel_ptr;
    HitablePtr          accel_ptr;
    Camera*             camera_ptr;
    Film*               film_ptr;
    std::vector<std::unique_ptr<Light>>* lights;

    OSL::PerThreadInfo* thread_info;
    OSL::ShadingContext* ctx;
    OSL::ShadingSystem* shadingsys;
    //KazenRenderServices rend;
    std::unordered_map<std::string, OSL::ShaderGroupRef>* shaders;
};

class NormalIntegrator : public Integrator {
public:
    NormalIntegrator();
    NormalIntegrator(Camera* cam_ptr, Film* flm_ptr);

    RGBSpectrum Li(const Ray& r) const override;
};

class AmbientOcclusionIntegrator : public Integrator {
public:
    AmbientOcclusionIntegrator();
    AmbientOcclusionIntegrator(Camera* cam_ptr, Film* flm_ptr);

    RGBSpectrum Li(const Ray& r) const override;
};

class PathIntegrator : public Integrator {
public:
    PathIntegrator();
    PathIntegrator(Camera* camera_ptr, Film* flm_ptr);

    RGBSpectrum Li(const Ray& r) const override;
};

class IntegratorFactory {
public:
    inline std::unique_ptr<Integrator> create() {
        return create_functor();
    }

    std::function<std::unique_ptr<Integrator>(void)> create_functor;
};

template<typename I>
std::unique_ptr<Integrator> generic_create(Camera* cam_ptr, Film* flm_ptr) {
    return std::make_unique<I>(cam_ptr, flm_ptr);
}