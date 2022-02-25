#pragma once

#include <array>
#include <functional>

#include <bsdf.h>

struct Diffuse {
    static float eval(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf);
    static float sample(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wi, float& pdf);
};

struct Phong {
    static float eval(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf);
    static float sample(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wi, float& pdf);
};

struct Emission {
    static float eval(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf);
    static float sample(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wi, float& pdf);
};

using eval_func = std::function<float(const void*, const OSL::ShaderGlobals&,
    const Vec3f&, float&)>;
using sample_func = std::function<float(const void*, const OSL::ShaderGlobals&,
    const Vec3f&, Vec3f&, float&)>;

// cpp 17 inlined constexpr variables will have external linkage will
// have only one copy among all included files
inline constexpr std::array<eval_func, 15> eval_functions {
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