#pragma once

#include <functional>
#include <memory>
#include <unordered_map>

#include <OpenImageIO/imageio.h>

#include "core/accel.h"
#include "core/camera.h"
#include "core/film.h"
#include "core/light.h"
#include "core/light_path.h"
#include "shading/renderservices.h"

class Scene;
class Camera;
class Film;

class Integrator {
public:
    Integrator();
    Integrator(Camera* cam_ptr, Film* flm_ptr, Recorder* rec);

    virtual void setup(Scene* scene);
    virtual RGBSpectrum Li(const Ray& r, const RecordContext& rctx) const = 0;

    HitablePtr          accel_ptr;
    Camera*             camera_ptr;
    Film*               film_ptr;
    std::vector<std::unique_ptr<Light>>* lights;

    Recorder*           recorder;
};

class NormalIntegrator : public Integrator {
public:
    NormalIntegrator();
    NormalIntegrator(Camera* cam_ptr, Film* flm_ptr, Recorder* rec);

    static std::unique_ptr<Integrator> create(Camera* cam_ptr, Film* flm_ptr, Recorder* rec) {
        return std::make_unique<NormalIntegrator>(cam_ptr, flm_ptr, rec);
    }

    RGBSpectrum Li(const Ray& r, const RecordContext& rctx) const override;
};

class AmbientOcclusionIntegrator : public Integrator {
public:
    AmbientOcclusionIntegrator();
    AmbientOcclusionIntegrator(Camera* cam_ptr, Film* flm_ptr, Recorder* rec);

    static std::unique_ptr<Integrator> create(Camera* cam_ptr, Film* flm_ptr, Recorder* rec) {
        return std::make_unique<AmbientOcclusionIntegrator>(cam_ptr, flm_ptr, rec);
    }

    RGBSpectrum Li(const Ray& r, const RecordContext& rctx) const override;
};

class OSLBasedIntegrator : public Integrator {
public:
    OSLBasedIntegrator() {}
    OSLBasedIntegrator(Camera* cam_ptr, Film* flm_ptr, Recorder* rec)
        : Integrator(cam_ptr, flm_ptr, rec)
    {}

    void setup(Scene* scene) override;

    OSL::ShadingSystem* shadingsys;
    std::unordered_map<std::string, OSL::ShaderGroupRef>* shaders;
    OSL::PerThreadInfo* thread_info;
    OSL::ShadingContext* ctx;
};

class WhittedIntegrator : public OSLBasedIntegrator {
public:
    WhittedIntegrator();
    WhittedIntegrator(Camera* camera_ptr, Film* flm_ptr, Recorder* rec);

    static std::unique_ptr<Integrator> create(Camera* cam_ptr, Film* flm_ptr, Recorder* rec) {
        return std::make_unique<WhittedIntegrator>(cam_ptr, flm_ptr, rec);
    }

    //void setup(Scene* scene) override;
    RGBSpectrum Li(const Ray& r, const RecordContext& rctx) const override;
};

class PathMatsIntegrator : public OSLBasedIntegrator {
public:
    PathMatsIntegrator();
    PathMatsIntegrator(Camera* camera_ptr, Film* flm_ptr, Recorder* rec);

    static std::unique_ptr<Integrator> create(Camera* cam_ptr, Film* flm_ptr, Recorder* rec) {
        return std::make_unique<PathMatsIntegrator>(cam_ptr, flm_ptr, rec);
    }

    //void setup(Scene* scene) override;
    RGBSpectrum Li(const Ray& r, const RecordContext& rctx) const override;
};

class PathIntegrator : public OSLBasedIntegrator {
public:
    PathIntegrator();
    PathIntegrator(Camera* camera_ptr, Film* flm_ptr, Recorder* rec);

    static std::unique_ptr<Integrator> create(Camera* cam_ptr, Film* flm_ptr, Recorder* rec) {
        return std::make_unique<PathIntegrator>(cam_ptr, flm_ptr, rec);
    }

    void setup(Scene* scene) override;
    RGBSpectrum Li(const Ray& r, const RecordContext& rctx) const override;
};

class OldPathIntegrator : public OSLBasedIntegrator {
public:
    OldPathIntegrator();
    OldPathIntegrator(Camera* camera_ptr, Film* flm_ptr, Recorder* rec);

    static std::unique_ptr<Integrator> create(Camera* cam_ptr, Film* flm_ptr, Recorder* rec) {
        return std::make_unique<OldPathIntegrator>(cam_ptr, flm_ptr, rec);
    }

    void setup(Scene* scene) override;
    RGBSpectrum Li(const Ray& r, const RecordContext& rctx) const override;
};

class IntegratorFactory {
public:
    inline std::unique_ptr<Integrator> create(Camera* cam_ptr, Film* flm_ptr, Recorder* rec=nullptr) {
        return create_functor(cam_ptr, flm_ptr, rec);
    }

    std::function<std::unique_ptr<Integrator>(Camera*, Film*, Recorder*)> create_functor;
};
