#pragma once

#include <array>
#include <functional>

#include <OSL/genclosure.h>

#include "core/sampling.h"
#include "shading/bsdf.h"
#include "shading/microfacet.h"

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

struct MicrofacetParams {
    OSL::ustring dist;
    OSL::Vec3 N;
    float alpha, eta;
    int refract;
};

struct MicrofacetAnisoParams {
    OSL::ustring dist;
    OSL::Vec3 N, U;
    float xalpha, yalpha, eta;
    int refract;
};

struct ReflectionParams {
    OSL::Vec3 N;
    float eta;
};

struct RefractionParams {
    OSL::Vec3 N;
    float eta;
};

struct KpDielectricParams {
    float int_ior = 1.5046f;
    float ext_ior = 1.000277f;
};

struct KpMicrofacetParams {
    float alpha = 0.1;
    float int_ior = 1.5046f;
    float ext_ior = 1.000277f;
    float kd = 0.5;
};

struct KpEmitterParams {
    float albedo;
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

struct Microfacet {
    static float eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static float sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_STRING_PARAM(MicrofacetParams, dist),
            CLOSURE_VECTOR_PARAM(MicrofacetParams, N),
            CLOSURE_FLOAT_PARAM(MicrofacetParams, alpha),
            CLOSURE_FLOAT_PARAM(MicrofacetParams, eta),
            CLOSURE_INT_PARAM(MicrofacetParams, refract),
            CLOSURE_FINISH_PARAM(MicrofacetParams)
        };

        shadingsys.register_closure("microfacet", MicrofacetID, params, nullptr, nullptr);
    }
};

// Make it template for different distribution later
//template <typename Dist, int Refract>
struct MicrofacetAniso {
    static float eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static float sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_STRING_PARAM(MicrofacetAnisoParams, dist),
            CLOSURE_VECTOR_PARAM(MicrofacetAnisoParams, N),
            CLOSURE_VECTOR_PARAM(MicrofacetAnisoParams, U),
            CLOSURE_FLOAT_PARAM(MicrofacetAnisoParams, xalpha),
            CLOSURE_FLOAT_PARAM(MicrofacetAnisoParams, yalpha),
            CLOSURE_FLOAT_PARAM(MicrofacetAnisoParams, eta),
            CLOSURE_INT_PARAM(MicrofacetAnisoParams, refract),
            CLOSURE_FINISH_PARAM(MicrofacetAnisoParams)
        };

        shadingsys.register_closure("microfacet", MicrofacetAnisoID, params, nullptr, nullptr);
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

struct Refraction {
    static float eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static float sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(RefractionParams, N),
            CLOSURE_FLOAT_PARAM(RefractionParams, eta),
            CLOSURE_FINISH_PARAM(RefractionParams)
        };

        shadingsys.register_closure("refraction", RefractionID, params, nullptr, nullptr);
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

struct KpMirror {
    static float eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static float sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_FINISH_PARAM(EmptyParams)
        };

        shadingsys.register_closure("kp_mirror", KpMirrorID, params, nullptr, nullptr);
    }
};

struct KpDielectric {
    static float eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static float sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_FLOAT_PARAM(KpDielectricParams, int_ior),
            CLOSURE_FLOAT_PARAM(KpDielectricParams, ext_ior),
            CLOSURE_FINISH_PARAM(KpDielectricParams)
        };

        shadingsys.register_closure("kp_dielectric", KpDielectricID, params, nullptr, nullptr);
    }
};

struct KpMicrofacet {
    static float eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static float sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_FLOAT_PARAM(KpMicrofacetParams, alpha),
            CLOSURE_FLOAT_PARAM(KpMicrofacetParams, int_ior),
            CLOSURE_FLOAT_PARAM(KpMicrofacetParams, ext_ior),
            CLOSURE_FLOAT_PARAM(KpMicrofacetParams, kd),
            CLOSURE_FINISH_PARAM(KpMicrofacetParams)
        };

        shadingsys.register_closure("kp_microfacet", KpMicrofacetID, params, nullptr, nullptr);
    }
};

struct KpEmitter {
    static float eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static float sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(KpEmitterParams, albedo),
            CLOSURE_FINISH_PARAM(KpEmitterParams)
        };

        shadingsys.register_closure("kp_emitter", KpEmitterID, params, nullptr, nullptr);
    }
};

using eval_func = std::function<float(const void*, const OSL::ShaderGlobals&,
    BSDFSample&)>;
using sample_func = std::function<float(const void*, const OSL::ShaderGlobals&,
    BSDFSample&)>;

// cpp 17 inlined constexpr variables will have external linkage will
// have only one copy among all included files
inline eval_func get_eval_func(ClosureID id) {
    static std::array<eval_func, 20> eval_functions {
        Diffuse::eval,
        Phong::eval,
        nullptr,
        nullptr,
        Reflection::eval,
        Refraction::eval,
        nullptr,
        nullptr,
        Microfacet::eval,
        MicrofacetAniso::eval,
        nullptr,
        Emission::eval, // emission
        nullptr,
        nullptr,
        nullptr,
        KpMirror::eval,
        KpDielectric::eval,
        KpMicrofacet::eval,
        KpEmitter::eval,
        nullptr
    };
    return eval_functions[id];
}

inline sample_func get_sample_func(ClosureID id) {
    static std::array<sample_func, 20> sample_functions {
        Diffuse::sample,
        Phong::sample,
        nullptr,
        nullptr,
        Reflection::sample,
        Refraction::sample,
        nullptr,
        nullptr,
        Microfacet::sample,
        MicrofacetAniso::sample,
        nullptr,
        Emission::sample, // emission
        nullptr,
        nullptr,
        nullptr,
        KpMirror::sample,
        KpDielectric::sample,
        KpMicrofacet::sample,
        KpEmitter::sample,
        nullptr
    };
    return sample_functions[id];
}