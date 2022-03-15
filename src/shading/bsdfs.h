#pragma once

#include <array>
#include <functional>

#include "shading/bsdf.h"

using OSL::TypeDesc;

struct Diffuse {
    static float eval(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf);
    static float sample(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wi, float& pdf);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(DiffuseParams, N),
            CLOSURE_FINISH_PARAM(DiffuseParams)
        };

        shadingsys.register_closure("diffuse", DiffuseID, params, nullptr, nullptr);
    }
};

struct Phong {
    static float eval(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf);
    static float sample(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wi, float& pdf);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(PhongParams, N),
            CLOSURE_VECTOR_PARAM(PhongParams, exponent),
            CLOSURE_FINISH_PARAM(PhongParams)
        };

        shadingsys.register_closure("phong", PhongID, params, nullptr, nullptr);
    }
};

struct Emission {
    static float eval(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf);
    static float sample(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wi, float& pdf);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_FINISH_PARAM(EmptyParams)
        };

        shadingsys.register_closure("emission", EmissionID, params, nullptr, nullptr);
    }
};

using eval_func = std::function<float(const void*, const OSL::ShaderGlobals&,
    const Vec3f&, float&)>;
using sample_func = std::function<float(const void*, const OSL::ShaderGlobals&,
    const Vec3f&, Vec3f&, float&)>;

// cpp 17 inlined constexpr variables will have external linkage will
// have only one copy among all included files
inline constexpr eval_func eval_functions[15] {
    Diffuse::eval,
    Phong::eval,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    Emission::eval, // emission
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

inline constexpr std::array<sample_func, 15> sample_functions {
    Diffuse::sample,
    Phong::sample,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    Emission::sample, // emission
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};