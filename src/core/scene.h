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

class Scene {
public:
    Scene();
    
    void parse_from_file(fs::path file_path);
    std::unique_ptr<Integrator> create_integrator(Sampler& sampler);

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