#pragma once

#include <array>
#include <functional>

#include <OSL/genclosure.h>

#include "shading/bsdf.h"

using OSL::TypeDesc;

struct EmptyParams      {};

struct DiffuseParams {
    OSL::Vec3 R;
    OSL::Vec3 N;
};

struct PhongParams {
    OSL::Vec3 R;
    OSL::Vec3 N;
    float exponent;
};

struct ReflectionParams {
    OSL::Vec3 N;
    float eta;
};

struct DielectricParams {
    float int_ior = 1.5046f;
    float ext_ior = 1.000277f;
};

struct Diffuse {
    static float eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static float sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(DiffuseParams, N),
            CLOSURE_FINISH_PARAM(DiffuseParams)
        };

        shadingsys.register_closure("diffuse", DiffuseID, params, nullptr, nullptr);
    }
};

struct Phong {
    static float eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static float sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(PhongParams, N),
            CLOSURE_VECTOR_PARAM(PhongParams, exponent),
            CLOSURE_FINISH_PARAM(PhongParams)
        };

        shadingsys.register_closure("phong", PhongID, params, nullptr, nullptr);
    }
};

struct Reflection {
    static float eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static float sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(ReflectionParams, N),
            CLOSURE_FLOAT_PARAM(ReflectionParams, eta),
            CLOSURE_FINISH_PARAM(ReflectionParams)
        };

        shadingsys.register_closure("reflection", ReflectionID, params, nullptr, nullptr);
    }
};

struct Emission {
    static float eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static float sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_FINISH_PARAM(EmptyParams)
        };

        shadingsys.register_closure("emission", EmissionID, params, nullptr, nullptr);
    }
};

struct Mirror {
    static float eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static float sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_FINISH_PARAM(EmptyParams)
        };

        shadingsys.register_closure("mirror", MirrorID, params, nullptr, nullptr);
    }
};

struct Dielectric {
    static float eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static float sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_FLOAT_PARAM(DielectricParams, int_ior),
            CLOSURE_FLOAT_PARAM(DielectricParams, ext_ior),
            CLOSURE_FINISH_PARAM(DielectricParams)
        };

        shadingsys.register_closure("dielectric", DielectricID, params, nullptr, nullptr);
    }
};

using eval_func = std::function<float(const void*, const OSL::ShaderGlobals&,
    BSDFSample&)>;
using sample_func = std::function<float(const void*, const OSL::ShaderGlobals&,
    BSDFSample&)>;

// cpp 17 inlined constexpr variables will have external linkage will
// have only one copy among all included files
inline eval_func get_eval_func(ClosureID id) {
    static std::array<eval_func, 17> eval_functions {
        Diffuse::eval,
        Phong::eval,
        nullptr,
        nullptr,
        Reflection::eval,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        Emission::eval, // emission
        nullptr,
        nullptr,
        nullptr,
        Mirror::eval,
        Dielectric::eval,
        nullptr
    };
    return eval_functions[id];
};

inline sample_func get_sample_func(ClosureID id) {
    static std::array<sample_func, 17> sample_functions {
        Diffuse::sample,
        Phong::sample,
        nullptr,
        nullptr,
        Reflection::sample,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        Emission::sample, // emission
        nullptr,
        nullptr,
        nullptr,
        Mirror::sample,
        Dielectric::sample,
        nullptr
    };
    return sample_functions[id];
};