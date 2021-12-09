#pragma once

#include <filesystem>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/accel.h"
#include "core/camera.h"
#include "core/film.h"
#include "core/integrator.h"
#include "core/light.h"
#include "core/material.h"
#include "core/shape.h"

namespace fs = std::filesystem;

class Scene {
public:
    Scene();
    
    void parse_from_file(fs::path file_path);

public:
    std::unique_ptr<Integrator> integrator;

public:
    // Make these fields public for now..
    std::unique_ptr<Film> film;
    std::unique_ptr<Camera> camera;
    std::unique_ptr<Hitable> accelerator;
    std::vector<std::shared_ptr<Hitable>> objects;
    std::vector<std::unique_ptr<Light>> lights;

    // OSL related
    KazenRenderServices rend;
    std::unique_ptr<OSL::ShadingSystem> shadingsys;
    std::unordered_map<std::string, OSL::ShaderGroupRef> shaders;
    OSL::ErrorHandler   errhandler;
};