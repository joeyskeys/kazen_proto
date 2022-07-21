#pragma once

#include <array>
#include <functional>

#include <OSL/genclosure.h>

#include "base/utils.h"
#include "core/sampling.h"
#include "shading/bsdf.h"
#include "shading/fresnel.h"
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
    OSL::Vec3 N, U;
    float xalpha, yalpha, eta;
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
    static float sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand);
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
    static float sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(PhongParams, N),
            CLOSURE_VECTOR_PARAM(PhongParams, exponent),
            CLOSURE_FINISH_PARAM(PhongParams)
        };

        shadingsys.register_closure("phong", PhongID, params, nullptr, nullptr);
    }
};

// Basically a replication of implementation in OpenShadingLanguage's testrender
// for now
// TODO : it's better to unify shading space to a z-up coordinate system
template <typename Dist, int Refract>
struct Microfacet {
    static float eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
        auto params = reinterpret_cast<const MicrofacetParams*>(data);
        auto wo = sample.wo;
        Vec3f wi = base::to_vec3(-sg.I);

        auto cos_theta_i = cos_theta(wi);
        auto cos_theta_o = cos_theta(wo);
        if constexpr (Refract == 0 || Refract == 2) {
            if (cos_theta_i > 0 && cos_theta_o > 0) {
                const Vec3f m = base::normalize(wi + wo);
                const float D = eval_D(m, params->xalpha, params->yalpha);
                const float lambda_o = eval_lambda(wo, params->xalpha, params->yalpha);
                const float lambda_i = eval_lambda(wi, params->xalpha, params->yalpha);
                const float G2 = eval_G2(lambda_o, lambda_i);
                const float G1 = eval_G1(lambda_i);

                const float Fr = fresnel_dielectric(base::dot(m, wi), params->eta);
                sample.pdf = (G1 * D * 0.25f) / cos_theta_i;
                float out = G2 / G1;
                if constexpr (Refract == 2) {
                    sample.pdf *= Fr;
                    return out;
                }
                else {
                    return out * Fr;
                }
            }
        }
        if constexpr (Refract == 1 || Refract == 2) {
            if (cos_theta_o < 0 && cos_theta_i > 0) {
                Vec3f ht = -(params->eta * wo + wi);
                if (params->eta < 1.f)
                    ht = -ht;
                Vec3f Ht = base::normalize(ht);
                const float cos_hi = base::dot(Ht, wi);
                const float Ft = 1.f - fresnel_dielectric(cos_hi, params->eta);
                if (Ft > 0) {
                    const float cos_ho = base::dot(Ht, wo);
                    const float cos_theta_m = cos_theta(Ht);
                    if (cos_theta_m < 0.f)
                        return 0;
                    const float Dt = eval_D(Ht, params->xalpha, params->yalpha);
                    const float lambda_o = eval_lambda(wo, params->xalpha, params->yalpha);
                    const float lambda_i = eval_lambda(wi, params->xalpha, params->yalpha);
                    const float G2 = eval_G2(lambda_i, lambda_o);
                    const float G1 = eval_G1(lambda_i);

                    float invHt2 = 1 / base::length_squared(ht);
                    sample.pdf = (fabsf(cos_ho * cos_hi) * square(params->eta) * (G1 * Dt) * invHt2) / cos_theta_i;
                    float out = G2 / G1;
                    if (Refract == 2) {
                        sample.pdf *= Ft;
                        return out;
                    }
                    else {
                        return out * Ft;
                    }
                }
            }
        }
        return sample.pdf = 0;
    }

    static float sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
        auto params = reinterpret_cast<const MicrofacetParams*>(data);
        const Vec3f wi = -sg.I;
        const float cos_ni = cos_theta(wi);
        if (!(cos_ni > 0)) return sample.pdf = 0;
        const Vec3f m = sample_slope(wi, params->xalpha, params->yalpha, Vec2f(rand[0], rand[1]));
        const float cos_mo = base::dot(m, wi);
        const float F = fresnel_dielectric(cos_mo, params->eta);
        if (Refract == 0 || (Refract == 2 && rand[2] < F)) {
            sample.wo = (2.f * cos_mo) * m - wi;
            const float D = eval_D(m, params->xalpha, params->yalpha);
            const float lambda_i = eval_lambda(wi, params->xalpha, params->yalpha);
            const float lambda_o = eval_lambda(sample.wo, params->xalpha, params->yalpha);

            const float G2 = eval_G2(lambda_i, lambda_o);
            const float G1 = eval_G1(lambda_i);

            sample.pdf = (G1 * D * 0.25f) / cos_ni;
            float out = G2 / G1;
            if constexpr (Refract == 2) {
                sample.pdf *= F;
                return out;
            }
            else {
                return F * out;
            }
        }
        else {
            float Ft = fresnel_refraction(sg.I, m, params->eta, sample.wo);
            const float cos_hi = base::dot(m, wi);
            const float cos_ho = base::dot(m, sample.wo);
            const float D = eval_D(m, params->xalpha, params->yalpha);
            const float lambda_o = eval_lambda(sample.wo, params->xalpha, params->yalpha);
            const float lambda_i = eval_lambda(wi, params->xalpha, params->yalpha);

            const float G2 = eval_G2(lambda_i, lambda_o);
            const float G1 = eval_G1(lambda_i);

            const Vec3f ht = -(params->eta * wi + sample.wo);
            const float invHt2 = 1.f / base::length(ht);

            sample.pdf = (fabsf(cos_hi * cos_ho) * square(params->eta) * (G1 * D) * invHt2) / fabsf(cos_theta(wi));
            float out = G2 / G1;
            if (Refract == 2) {
                sample.pdf *= Ft;
                return out;
            }
            else
                return Ft * out;
        }
        return sample.pdf = 0;
    }

    /*
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_STRING_PARAM(MicrofacetParams, dist),
            CLOSURE_VECTOR_PARAM(MicrofacetParams, N),
            CLOSURE_VECTOR_PARAM(MicrofacetParams, U),
            CLOSURE_FLOAT_PARAM(MicrofacetParams, xalpha),
            CLOSURE_FLOAT_PARAM(MicrofacetParams, yalpha),
            CLOSURE_FLOAT_PARAM(MicrofacetParams, eta),
            CLOSURE_INT_PARAM(MicrofacetParams, refract),
            CLOSURE_FINISH_PARAM(MicrofacetParams)
        };

        shadingsys.register_closure("microfacet", MicrofacetID, params, nullptr, nullptr);
    }
    */

private:
    inline static float eval_D(const Vec3f Hr, const float xalpha, const float yalpha) {
        float cos_theta_m = cos_theta(Hr);
        if (cos_theta_m > 0) {
            float cos_phi_2_st2 = square(Hr.x() / xalpha);
            float sin_phi_2_st2 = square(Hr.z() / yalpha);
            float cos_theta_m2 = square(cos_theta_m);
            float cos_theta_m4 = square(cos_theta_m2);
            float tan_theta_m2 = (cos_phi_2_st2 + sin_phi_2_st2) / cos_theta_m2;
            return Dist::D(tan_theta_m2) / (xalpha * yalpha * cos_theta_m4);
        }
        return 0;
    }

    inline static float eval_lambda(const Vec3f& w, const float xalpha, const float yalpha) {
        float cos_theta_2 = square(cos_theta(w));
        float cos_phi_2_st2 = square(w.x() * xalpha);
        float sin_phi_2_st2 = square(w.z() * yalpha);
        return Dist::lambda(cos_theta_2 / (cos_phi_2_st2 + sin_phi_2_st2));
    }

    static float eval_G2(const float lambda_i, const float lambda_o) {
        return 1 / (lambda_i + lambda_o + 1);
    }

    static float eval_G1(const float lambda_v) {
        return 1 / (lambda_v + 1);
    }

    inline static Vec3f sample_slope(const Vec3f wi, const float xalpha, const float yalpha, Vec2f sample) {
        // Stretch by alpha values
        Vec3f stretched_wi = wi;
        stretched_wi[0] *= xalpha;
        stretched_wi[2] *= yalpha;
        stretched_wi = base::normalize(stretched_wi);

        // Figure out angles for the incoming vector
        float cos_theta_i = std::max(0.f, cos_theta(stretched_wi));
        float cos_phi = 1;
        float sin_phi = 0;
        // Special case gets phi 0
        if (cos_theta_i < 0.999999f) {
            float invnorm = 1 / base::length(stretched_wi);
            cos_phi = stretched_wi.x() * invnorm;
            sin_phi = stretched_wi.z() * invnorm;
        }

        Vec2f slope = Dist::sample_slope(cos_theta_i, sample);

        // Rotate and unstretch
        Vec2f s(cos_phi * slope.x() - sin_phi * slope.y(),
                sin_phi * slope.x() + cos_phi * slope.y());
        s[0] *= xalpha;
        s[1] *= yalpha;

        float mlen = sqrtf(s.x() * s.x() + s.y() * s.y() + 1);
        Vec3f m(fabsf(s.x()) < mlen ? -s.x() / mlen : 1.f,
                1.f / mlen,
                fabsf(s.y()) < mlen ? -s.y() / mlen : 1.f);
        return m;
    }
};

using MicrofacetGGXRefl = Microfacet<GGXDist, 0>;
using MicrofacetGGXRefr = Microfacet<GGXDist, 1>;
using MicrofacetGGXBoth = Microfacet<GGXDist, 2>;
using MicrofacetBeckmannRefl = Microfacet<BeckmannDist, 0>;
using MicrofacetBeckmannRefr = Microfacet<BeckmannDist, 1>;
using MicrofacetBeckmannBoth = Microfacet<BeckmannDist, 2>;

/*
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
*/

struct Reflection {
    static float eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static float sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand);
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
    static float sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand);
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
    static float sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_FINISH_PARAM(EmptyParams)
        };

        shadingsys.register_closure("emission", EmissionID, params, nullptr, nullptr);
    }
};

struct KpMirror {
    static float eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static float sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_FINISH_PARAM(EmptyParams)
        };

        shadingsys.register_closure("kp_mirror", KpMirrorID, params, nullptr, nullptr);
    }
};

struct KpDielectric {
    static float eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample);
    static float sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand);
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
    static float sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand);
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
    static float sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_FLOAT_PARAM(KpEmitterParams, albedo),
            CLOSURE_FINISH_PARAM(KpEmitterParams)
        };

        shadingsys.register_closure("kp_emitter", KpEmitterID, params, nullptr, nullptr);
    }
};

using eval_func = std::function<float(const void*, const OSL::ShaderGlobals&,
    BSDFSample&)>;
using sample_func = std::function<float(const void*, const OSL::ShaderGlobals&,
    BSDFSample&, const Vec3f)>;

// cpp 17 inlined constexpr variables will have external linkage will
// have only one copy among all included files
inline eval_func get_eval_func(ClosureID id) {
    static std::array<eval_func, 25> eval_functions {
        Diffuse::eval,
        Phong::eval,
        nullptr,
        nullptr,
        Reflection::eval,
        Refraction::eval,
        nullptr,
        nullptr,
        //Microfacet::eval,
        nullptr,
        MicrofacetGGXRefl::eval,
        MicrofacetGGXRefr::eval,
        MicrofacetGGXBoth::eval,
        MicrofacetBeckmannRefl::eval,
        MicrofacetBeckmannRefr::eval,
        MicrofacetBeckmannBoth::eval,
        //MicrofacetAniso::eval,
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
    static std::array<sample_func, 25> sample_functions {
        Diffuse::sample,
        Phong::sample,
        nullptr,
        nullptr,
        Reflection::sample,
        Refraction::sample,
        nullptr,
        nullptr,
        //Microfacet::sample,
        nullptr,
        MicrofacetGGXRefl::sample,
        MicrofacetGGXRefr::sample,
        MicrofacetGGXBoth::sample,
        MicrofacetBeckmannRefl::sample,
        MicrofacetBeckmannRefr::sample,
        MicrofacetBeckmannBoth::sample,
        //MicrofacetAniso::sample,
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