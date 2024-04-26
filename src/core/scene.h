#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include <pugixml.hpp>

#include "base/traits.h"
#include "core/accel.h"
#include "core/camera.h"
#include "core/film.h"
#include "core/integrator.h"
#include "core/light.h"
#include "core/material.h"
#include "core/shape.h"
#include "kernel/types.h"
#include "shading/compiler.h"

namespace fs = std::filesystem;

enum class AcceleratorType {
    BVH,
    Embree,
    Optix
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

    inline void set_film(uint w, uint h, uint xres, uint yres, const std::string& f) {
        film.reset(new Film{w, h, xres, yres, f});
        film->generate_tiles();
    }

    inline void set_camera(const Vec3f& p, const Vec3f& l, const Vec3f& u,
        const float near_plane=1, const float far_plane=1000,
        const float fov=60)
    {
        if (!film)
            throw std::runtime_error("Film not set yet");
        *camera = Camera(p, l, u, near_plane, far_plane, fov, film.get(), true);
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
        const std::string& name, const std::string& shader_name, bool is_light=false)
    {
        Transform trans{world};
        auto obj_ptr = std::make_shared<Sphere>(trans, name, objects.size(),
            base::concat(p, r), shader_name);
        accelerator->add_sphere(obj_ptr);
    }

    inline void add_triangle(const Mat4f& world, const Vec3f& a, const Vec3f& b,
        const Vec3f& c, const std::string& name, const std::string& shader_name, bool is_light=false)
    {
        Transform trans{world};
        auto obj_ptr = std::make_shared<Triangle>(trans, a, b, c, name, shader_name);
        accelerator->add_triangle(obj_ptr);
    }

    inline void add_quad(const Mat4f& world, const Vec3f& c, const Vec3f& d,
        const Vec3f& u, const float w, const float h, const std::string& name,
        const std::string& shader_name, bool is_light=false)
    {
        Transform trans{world};
        auto obj_ptr = std::make_shared<Quad>(trans, c, d, u, w, h, name, shader_name);
        accelerator->add_quad(obj_ptr);
    }

    inline void add_mesh(const Mat4f& world, const std::vector<Vec3f>& vs,
        const std::vector<Vec3f>& ns, const std::vector<Vec2f>& ts,
        const std::vector<Vec3i>& idx, const std::string& name,
        const std::string& shader_name, bool is_light=false)
    {
        Transform trans{world};
        auto obj_ptr = std::make_shared<TriangleMesh>(trans, vs, ns, ts,
            idx, name, shader_name, is_light);
        obj_ptr->setup_dpdf();
        accelerator->add_trianglemesh(obj_ptr);
        if (is_light) {
            auto light = std::make_unique<GeometryLight>(lights.size(),
                obj_ptr->shader_name, obj_ptr);
            obj_ptr->light = light.get();
            lights.emplace_back(std::move(light));
        }
    }

    inline void build_bvh(const std::vector<std::string>& names) {
        accelerator->build(names);
    }

    inline void add_point_light(const RGBSpectrum& r, const Vec3f& p,
        const std::string& sname)
    {
        auto lgt_ptr = std::make_unique<PointLight>(lights.size());
        lgt_ptr->radiance = r;
        lgt_ptr->position = p;
        lgt_ptr->shader_name = sname;
        lights.emplace_back(std::move(lgt_ptr));
    }

    inline void set_shader_search_path(const std::string& path) {
        shadingsys->attribute("searchpath:shader", path.c_str());
    }

    void begin_shader_group(const std::string& name);
    void end_shader_group();
    bool load_oso_shader(const std::string& type, const std::string& name,
        const std::string& layer, const std::string& lib_path);
    void connect_shader(const std::string&, const std::string&, const std::string&,
        const std::string&);

    template <typename T>
    void set_shader_param(const std::string& name, const T& value) {
        if constexpr (is_bool_v<T> || is_int_v<T>)
            shadingsys->Parameter(*current_shader_group, name,
                OSL::TypeDesc::TypeInt, &value);
        else if constexpr (is_float_v<T>)
            shadingsys->Parameter(*current_shader_group, name,
                OSL::TypeDesc::TypeFloat, &value);
        else if constexpr (is_vec3f_v<T>)
            shadingsys->Parameter(*current_shader_group, name,
                OSL::TypeDesc::TypeVector, &value);
        else if constexpr (is_vec4f_v<T>)
            shadingsys->Parameter(*current_shader_group, name,
                OSL::TypeDesc::TypeString, &value);
        else if constexpr (is_str_v<T> || is_ustr_v<T>)
            shadingsys->Parameter(*current_shader_group, name,
                OSL::TypeDesc::TypeString, &value);
        else
            throw std::runtime_error(fmt::format(
                "Type {} is not supported for OSL parameter", typeid(T).name()));
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

private:
    // This is a temprory variable for use of shader group begin&end
    OSL::ShaderGroupRef current_shader_group;
};

class SceneGPU {
public:
    SceneGPU(bool default_pipeline = true);
    virtual ~SceneGPU();

    void create_default_pipeline();
    void set_film(uint32_t w, uint32_t h, const std::string& out);
    void set_camera(const Vec3f& p, const Vec3f& l, const Vec3f& u,
        const float ratio, const float near_plane = 1,
        const float far_plane = 1000, const float fov = 60.f);

    void add_mesh(const Mat4f& world, const std::vector<Vec3f>& vs,
        const std::vector<Vec3f>& ns, const std::vector<Vec2f>& ts,
        const std::vector<Vec3i>& idx, const std::string& name,
        const std::string& shader_name, bool is_light = false);
    void build_bvh(const std::vector<std::string>& names);

    // Considering inheriting or something else to share the OSL stuffs
    // since it's common
    inline void set_shader_search_path(const std::string& path) {
        shadingsys->attribute("searchpath:shader", path.c_str());
    }

    void begin_shader_group(const std::string& name);
    void end_shader_group();
    bool load_oso_shader(const std::string&, const std::string&, const std::string&,
        const std::string&);
    void connect_shader(const std::string&, const std::string&, const std::string&,
        const std::string&);

    template <typename T>
    void set_shader_param(const std::string& name, const T& value) {
        if constexpr (is_bool_v<T> || is_int_v<T>)
            shadingsys->Parameter(*current_shader_group, name,
                OSL::TypeDesc::TypeInt, &value);
        else if constexpr (is_float_v<T>)
            shadingsys->Parameter(*current_shader_group, name,
                OSL::TypeDesc::TypeFloat, &value);
        else if constexpr (is_vec3f_v<T>)
            shadingsys->Parameter(*current_shader_group, name,
                OSL::TypeDesc::TypeVector, &value);
        else if constexpr (is_vec4f_v<T>)
            shadingsys->Parameter(*current_shader_group, name,
                OSL::TypeDesc::TypeString, &value);
        else if constexpr (is_str_v<T> || is_ustr_v<T>)
            shadingsys->Parameter(*current_shader_group, name,
                OSL::TypeDesc::TypeString, &value);
        else
            throw std::runtime_error(fmt::format(
                "Type {} is not supported for OSL parameter", typeid(T).name()));
    }

private:
    OptixDeviceContext          ctx;
    OptixPipeline               ppl = nullptr;
    OptixShaderBindingTable     sbt = {};
    std::unique_ptr<OptixAccel> accel = nullptr;
    Params                      params = {};
    std::vector<std::unique_ptr<Light>> lights;
    std::string                 output = "./test.png";

public:
    // OSL related
    ShaderCompiler compiler;
    KazenRenderServices rend;
    std::unique_ptr<OSL::ShadingSystem> shadingsys;
    std::unordered_map<std::string, OSL::ShaderGroupRef> shaders;
    OSL::ShaderGroupRef background_shader;
    OSL::ErrorHandler   errhandler;

private:
    // This is a temprory variable for use of shader group begin&end
    OSL::ShaderGroupRef current_shader_group;
};