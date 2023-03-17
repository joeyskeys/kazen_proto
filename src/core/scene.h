#pragma once

#include <filesystem>
#include <memory>
#include <unordered_map>
#include <vector>

#include <pugixml.hpp>

#include "core/accel.h"
#include "core/camera.h"
#include "core/film.h"
#include "core/integrator.h"
#include "core/light.h"
#include "core/material.h"
#include "core/shape.h"
#include "shading/compiler.h"

namespace fs = std::filesystem;

enum class AcceleratorType {
    BVH,
    Embree
};

enum class IntegratorType {
    NormalIntegrator,
    AmbientOcclusionIntegrator,
    WhittedIntegrator,
    PathMatsIntegrator,
    PathEmsIntegrator,
    PathIntegrator
};

class Scene {
public:
    Scene();
    
    void parse_from_file(fs::path file_path);
    std::unique_ptr<Integrator> create_integrator(Sampler& sampler);

    inline void set_film(uint w, uint h, const std::string& f) {
        *film = Film(w, h, f);
    }

    inline void set_camera(const Vec3f& p, const Vec3f& l, const Vec3f& u,
        const float near_plane=1, const float far_plane=1000,
        const float fov=60, Film* const film=nullptr)
    {
        *camera = Camera(p, l, u, near_plane, far_plane, fov, film);
    }

    inline void set_accelerator(AcceleratorType type) {
        switch (type) {
            case AcceleratorType::BVH: {
                accelerator.reset(new BVHAccel(&objects));
                break;
            }

            case AcceleratorType::Embree: {
                accelerator.reset(new EmbreeAccel(&objects));
                break;
            }

            default: {
                accelerator.reset(new EmbreeAccel(&objects));
                break;
            }
        }
    }

    inline void set_integrator(IntegratorType type) {
        switch (type) {
            case IntegratorType::NormalIntegrator: {
                integrator_fac.create_functor = &NormalIntegrator::create;
                break;
            }

            case IntegratorType::AmbientOcclusionIntegrator: {
                integrator_fac.create_functor = &AmbientOcclusionIntegrator::create;
                break;
            }

            case IntegratorType::WhittedIntegrator: {
                integrator_fac.create_functor = &WhittedIntegrator::create;
                break;
            }

            case IntegratorType::PathMatsIntegrator: {
                integrator_fac.create_functor = &PathMatsIntegrator::create;
                break;
            }

            case IntegratorType::PathEmsIntegrator: {
                integrator_fac.create_functor = &PathEmsIntegrator::create;
                break;
            }

            case IntegratorType::PathIntegrator: {
                integrator_fac.create_functor = &PathIntegrator::create;
                break;
            }

            default: {
                integrator_fac.create_functor = &PathIntegrator::create;
                break;
            }
        }
    }

    inline void add_sphere(const Mat4f& world, const Vec3f& p, const float r,
        const std::string& shader_name, bool is_light=false)
    {
        Transform trans{world};
        auto obj_ptr = std::make_shared<Sphere>(trans, objects.size(),
            Vec4f{p, r}, shader_name);
        accelerator->add_sphere(obj_ptr);
    }

    inline void add_triangle(const Mat4f& world, const Vec3f& a, const Vec3f& b,
        const Vec3f& c, const std::string& shader_name, bool is_light=false)
    {
        Transform trans{world};
        auto obj_ptr = std::make_shared<Triangle>(trans, a, b, c, shader_name);
        accelerator->add_triangle(obj_ptr);
    }

    inline void add_quad(const Mat4f& world, const Vec3f& c, const Vec3f& d,
        const Vec3f& u, const float w, const float h, const std::string& shader_name,
        bool is_light=false)
    {
        Transform trans{world};
        auto obj_ptr = std::make_shared<Quad>(trans, c, d, u, w, h, shader_name);
        accelerator->add_quad(obj_ptr);
    }

    inline void add_mesh(const Mat4f& world, const std::vector<Vec3f>& vs,
        const std::vector<Vec3f>& ns, const std::vector<Vec2f>& ts,
        const std::vector<Vec3i>& idx, const std::string& shader_name,
        bool is_light=false)
    {
        Transform trans{world};
        auto obj_ptr = std::make_shared<TriangleMesh>(trans, vs, ns, ts,
            idx, shader_name);
        accelerator->add_trianglemesh(obj_ptr);
        if (is_light) {
            auto light = std::make_unique<GeometryLight>(lights.size(),
                obj_ptr->shader_name, obj_ptr);
            obj_ptr->light = light.get();
            lights.emplace_back(std::move(light));
        }
    }

    inline void add_point_light(const RGBSpectrum& r, const Vec3f& p) {
        auto lgt_ptr = std::make_unique<PointLight>(lights.size());
        lgt_ptr->radiance = r;
        lgt_ptr->position = p;
        lights.emplace_back(std::move(lgt_ptr));
    }

private:
    bool process_shader_node(const pugi::xml_node& node, OSL::ShaderGroupRef shader_group);

public:
    // Make these fields public for now..
    std::unique_ptr<Film> film;
    std::unique_ptr<Camera> camera;
    std::unique_ptr<Accelerator> accelerator;
    std::vector<std::shared_ptr<Hitable>> objects;
    std::vector<std::unique_ptr<Light>> lights;
    IntegratorFactory integrator_fac;
    fs::path working_dir;

    Recorder recorder;

    // OSL related
    ShaderCompiler compiler;
    KazenRenderServices rend;
    std::unique_ptr<OSL::ShadingSystem> shadingsys;
    std::unordered_map<std::string, OSL::ShaderGroupRef> shaders;
    OSL::ShaderGroupRef background_shader;
    OSL::ErrorHandler   errhandler;
};