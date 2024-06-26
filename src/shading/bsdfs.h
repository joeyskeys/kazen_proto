#pragma once

#include <array>
#include <functional>

#include <OSL/genclosure.h>

#include "base/utils.h"
#include "core/sampler.h"
#include "core/sampling.h"
#include "shading/bsdf.h"
#include "shading/context.h"
#include "shading/fresnel.h"
#include "shading/microfacet.h"

using OSL::TypeDesc;

struct EmptyParams      {};

struct DiffuseParams {
    OSL::Vec3 N;
};

struct PhongParams {
    OSL::Vec3 R;
    OSL::Vec3 N;
    float exponent;
};

struct OrenNayarParams {
    OSL::Vec3 N;
    float sigma;
};

struct WardParams {
    OSL::Vec3 N;
    OSL::Vec3 T;
    float xalpha, yalpha;
};

struct MicrofacetParams {
    OSL::ustring dist;
    OSL::Vec3 N, U;
    float xalpha, yalpha, eta;
    int refract;
};

struct SubsurfaceParams {
    OSL::Vec3   N;
    OSL::Vec3   albedo;
    OSL::Vec3   transmission_color;
    float       transmission_depth, anisotropy;
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

struct KpRoughParams {
    OSL::Vec3 N;
    float xalpha, yalpha, eta, f;
    OSL::ustring dist;
};

struct KpPrincipleRetroParams {
    OSL::Vec3 N;
    float roughness;
};

struct KpPrincipleFakeSSParams {
    float roughness;
};

struct KpPrincipleSheenParams {
    OSL::Vec3 N;
    float sheen;
};

// No need to create parameter struct for each closure
// Perhaps change parameter struct to a naming style like
// ParamVFFFF indicating components containing one vector and four floats
// will be more convinient which can be shared between multiple closures.
// But the naming of parameter will be kinda confusing...
struct KpPrincipleSpecularParams {
    OSL::Vec3 F0;
    OSL::Vec3 N;
    float xalpha, yalpha, eta, metallic;
};

struct KpPrincipleClearcoatParams {
    OSL::Vec3 N;
    float roughness;
};

struct KpPrincipleBSSRDFParams {
    OSL::Vec3 scatter_distance;
};

struct Diffuse {
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(DiffuseParams, N),
            CLOSURE_FINISH_PARAM(DiffuseParams)
        };

        shadingsys.register_closure("diffuse", DiffuseID, params, nullptr, nullptr);
    }
};

struct Phong {
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(PhongParams, N),
            CLOSURE_VECTOR_PARAM(PhongParams, exponent),
            CLOSURE_FINISH_PARAM(PhongParams)
        };

        shadingsys.register_closure("phong", PhongID, params, nullptr, nullptr);
    }
};

struct OrenNayar {
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(OrenNayarParams, N),
            CLOSURE_FLOAT_PARAM(OrenNayarParams, sigma),
            CLOSURE_FINISH_PARAM(OrenNayarParams)
        };

        shadingsys.register_closure("oren_nayar", OrenNayarID, params, nullptr, nullptr);
    }
};

struct Ward {
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(WardParams, N),
            CLOSURE_VECTOR_PARAM(WardParams, T),
            CLOSURE_FLOAT_PARAM(WardParams, xalpha),
            CLOSURE_FLOAT_PARAM(WardParams, yalpha),
            CLOSURE_FINISH_PARAM(WardParams)
        };

        shadingsys.register_closure("ward", WardID, params, nullptr, nullptr);        
    }
};

/*
// Basically a replication of implementation in OpenShadingLanguage's testrender
// for now
// Seems this impl isn't helping much...
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
                //const float Fr = fresnel_refl_dielectric(params->eta, base::dot(m, wi));
                sample.pdf = (G1 * D * 0.25f) / cos_theta_i;
                float out = D * G2 / G1;
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

    static float sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, Sampler* rng) {
        auto params = reinterpret_cast<const MicrofacetParams*>(data);
        const Vec3f wi = -sg.I;
        const float cos_ni = cos_theta(wi);
        auto rand = rng->random3f();
        if (!(cos_ni > 0)) return sample.pdf = 0;
        const Vec3f m = sample_slope(wi, params->xalpha, params->yalpha, Vec2f(rand[0], rand[1]));
        const float cos_mo = base::dot(m, wi);
        const float F = fresnel_dielectric(cos_mo, params->eta);
        if (Refract == 0 || (Refract == 2 && rand[2] < F)) {
            sample.wo = reflect(wi, m);
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
            float Ft = fresnel_refraction(base::from_osl_vec3(sg.I), m, params->eta, sample.wo);
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

private:
    inline static float eval_D(const Vec3f m, const float xalpha, const float yalpha) {
        float cos_theta_m = cos_theta(m);
        if (cos_theta_m > 0) {
            //float cos_phi_2_st2 = square(Hr.x() / xalpha);
            //float sin_phi_2_st2 = square(Hr.z() / yalpha);
            float cos_theta_m2 = square(cos_theta_m);
            float cos_theta_m4 = square(cos_theta_m2);
            //float tan_theta_m2 = (cos_phi_2_st2 + sin_phi_2_st2) * (1 - cos_theta_m2) / cos_theta_m2;
            //float tan_theta_m2 = (cos_phi_2_st2 + sin_phi_2_st2) / cos_theta_m2;
            const float tan_theta_m2 = (1.f - cos_theta_m2) / cos_theta_m2;
            const float A = stretched_roughness(m, xalpha, yalpha);
            return Dist::D(tan_theta_m2 * A) / (xalpha * yalpha * cos_theta_m4);
        }
        return 0;
    }

    inline static float eval_lambda(const Vec3f& w, const float xalpha, const float yalpha) {
        float cos_theta_2 = square(cos_theta(w));
        float cos_phi_2_st2 = square(w.x() * xalpha);
        float sin_phi_2_st2 = square(w.z() * yalpha);
        //return Dist::lambda(cos_theta_2 / (cos_phi_2_st2 + sin_phi_2_st2));
        return Dist::lambda(1 / (tan_2_theta(w) * (cos_phi_2_st2 + sin_phi_2_st2)));
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
            float invnorm = 1 / sqrtf(square(stretched_wi[0]) + square(stretched_wi[2]));
            cos_phi = stretched_wi.x() * invnorm;
            sin_phi = stretched_wi.z() * invnorm;
        }

        Vec2f slope = Dist::sample_slope(cos_theta_i, sample);

        // Rotate and unstretch
        Vec2f s(cos_phi * slope.x() - sin_phi * slope.y(),
                sin_phi * slope.x() + cos_phi * slope.y());

        Vec3f m{-s[0] * xalpha, 1.f, -s[1] * yalpha};

        return normalize(m);
    }
};

using MicrofacetGGXRefl = Microfacet<GGXDist, 0>;
using MicrofacetGGXRefr = Microfacet<GGXDist, 1>;
using MicrofacetGGXBoth = Microfacet<GGXDist, 2>;
using MicrofacetBeckmannRefl = Microfacet<BeckmannDist, 0>;
using MicrofacetBeckmannRefr = Microfacet<BeckmannDist, 1>;
using MicrofacetBeckmannBoth = Microfacet<BeckmannDist, 2>;
*/

// Since now we are using KpGloss & KpGlass to represent the actual Microfacet
// closure, we don't need a explicit Microfacet struct.
// Just convert it to KpGloss & KpGlass in process_closures function.

struct Reflection {
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
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
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(RefractionParams, N),
            CLOSURE_FLOAT_PARAM(RefractionParams, eta),
            CLOSURE_FINISH_PARAM(RefractionParams)
        };

        shadingsys.register_closure("refraction", RefractionID, params, nullptr, nullptr);
    }
};

struct Transparent {
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_FINISH_PARAM(EmptyParams)
        };

        shadingsys.register_closure("transparent", TransparentID, params, nullptr, nullptr);
    }
};

struct Translucent {
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_FINISH_PARAM(EmptyParams)
        };

        shadingsys.register_closure("translucent", TranslucentID, params, nullptr, nullptr);
    }
};

struct Emission {
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_FINISH_PARAM(EmptyParams)
        };

        shadingsys.register_closure("emission", EmissionID, params, nullptr, nullptr);
    }
};

struct Background {
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_FINISH_PARAM(EmptyParams)
        };

        shadingsys.register_closure("background", BackgroundID, params, nullptr, nullptr);
    }
};

struct KpMirror {
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_FINISH_PARAM(EmptyParams)
        };

        shadingsys.register_closure("kp_mirror", KpMirrorID, params, nullptr, nullptr);
    }
};

struct KpDielectric {
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
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
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
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
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_FLOAT_PARAM(KpEmitterParams, albedo),
            CLOSURE_FINISH_PARAM(KpEmitterParams)
        };

        shadingsys.register_closure("kp_emitter", KpEmitterID, params, nullptr, nullptr);
    }
};

struct KpGloss {
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(KpRoughParams, N),
            CLOSURE_FLOAT_PARAM(KpRoughParams, xalpha),
            CLOSURE_FLOAT_PARAM(KpRoughParams, yalpha),
            CLOSURE_FLOAT_PARAM(KpRoughParams, eta),
            CLOSURE_FLOAT_PARAM(KpRoughParams, f),
            CLOSURE_FINISH_PARAM(KpRoughParams)
        };

        shadingsys.register_closure("kp_gloss", KpGlossID, params, nullptr, nullptr);
    }
};

struct KpGlass {
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(KpRoughParams, N),
            CLOSURE_FLOAT_PARAM(KpRoughParams, xalpha),
            CLOSURE_FLOAT_PARAM(KpRoughParams, yalpha),
            CLOSURE_FLOAT_PARAM(KpRoughParams, eta),
            CLOSURE_FLOAT_PARAM(KpRoughParams, f),
            CLOSURE_FINISH_PARAM(KpRoughParams)
        };

        shadingsys.register_closure("kp_glass", KpGlassID, params, nullptr, nullptr);
    }

private:
    template <typename MDF>
    static inline float eval_reflection(const MDF& mdf, const Vec3f& wo, const Vec3f& m,
        const float F)
    {
        const float denom = std::abs(4.f * mdf.wi[1] * wo[1]);
        if (denom == 0.f)
            return 0.f;

        const float D = mdf.D(m);
        const float G = mdf.G(wo);
        return F * D * G / denom;
    }

    template <typename MDF>
    static inline float reflection_pdf(const MDF& mdf, const Vec3f& m, const float cos_mi)
    {
        if (cos_mi == 0.f)
            return 0.f;

        const float jacobian = 1.f / (4.f * std::abs(cos_mi));
        return jacobian * mdf.pdf(m);
    }

    template <typename MDF>
    static inline float eval_refraction(const MDF& mdf, const float eta, const Vec3f& wo,
        const Vec3f& m, const float T)
    {
        if (mdf.wi[1] == 0.f || wo[1] == 0.f)
            return 0.f;

        const float cos_mi = base::dot(m, mdf.wi);
        const float cos_mo = base::dot(m, wo);
        const float c = std::abs((cos_mi * cos_mo) / (mdf.wi[1] * wo[1]));

        float denom = cos_mi + eta * cos_mo;
        denom = square(denom);
        if (std::abs(denom) < 1.0e-6f)
            return 0.f;

        const float D = mdf.D(m);
        const float G = mdf.G(wo);

        return c * D * G * T * square(eta) / square(denom);
    }

    template <typename MDF>
    static inline float refraction_pdf(const MDF& mdf, const Vec3f& wo, const Vec3f& m,
        const float eta)
    {
        auto cos_mo = base::dot(m, wo);
        auto cos_mi = base::dot(m, mdf.wi);
        auto denom = cos_mi + eta * cos_mo;
        if (std::abs(denom) < 1.0e-6f)
            return 0.f;

        auto jacobian = std::abs(cos_mo) * square(eta / denom);
        return jacobian * mdf.pdf(m);
    }
};

struct KpPrincipleDiffuse {
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(DiffuseParams, N),
            CLOSURE_FINISH_PARAM(DiffuseParams)
        };

        shadingsys.register_closure("kp_principle_diffuse", KpPrincipleDiffuseID, params, nullptr, nullptr);
    }
};

struct KpPrincipleRetro {
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(KpPrincipleRetroParams, N),
            CLOSURE_FLOAT_PARAM(KpPrincipleRetroParams, roughness),
            CLOSURE_FINISH_PARAM(KpPrincipleRetroParams)
        };

        shadingsys.register_closure("kp_principle_retro", KpPrincipleRetroID, params, nullptr, nullptr);
    }
};

struct KpPrincipleFakeSS {
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_FLOAT_PARAM(KpPrincipleFakeSSParams, roughness),
            CLOSURE_FINISH_PARAM(KpPrincipleFakeSSParams)
        };

        shadingsys.register_closure("kp_principle_fakess", KpPrincipleFakeSSID, params, nullptr, nullptr);
    }
};

struct KpPrincipleSheen {
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(KpPrincipleSheenParams, N),
            CLOSURE_FINISH_PARAM(KpPrincipleSheenParams)
        };

        shadingsys.register_closure("kp_principle_sheen", KpPrincipleSheenID, params, nullptr, nullptr);
    }
};

struct KpPrincipleSpecularReflection {
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_COLOR_PARAM(KpPrincipleSpecularParams, F0),
            CLOSURE_VECTOR_PARAM(KpPrincipleSpecularParams, N),
            CLOSURE_FLOAT_PARAM(KpPrincipleSpecularParams, xalpha),
            CLOSURE_FLOAT_PARAM(KpPrincipleSpecularParams, yalpha),
            CLOSURE_FLOAT_PARAM(KpPrincipleSpecularParams, eta),
            CLOSURE_FLOAT_PARAM(KpPrincipleSpecularParams, metallic),
            CLOSURE_FINISH_PARAM(KpPrincipleSpecularParams)
        };

        shadingsys.register_closure("kp_principle_specular_reflection", KpPrincipleSpecularReflectionID, params, nullptr, nullptr);
    }
};

struct KpPrincipleClearcoat {
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(KpPrincipleClearcoatParams, N),
            CLOSURE_FLOAT_PARAM(KpPrincipleClearcoatParams, roughness),
            CLOSURE_FINISH_PARAM(KpPrincipleClearcoatParams)
        };

        shadingsys.register_closure("kp_principle_clearcoat", KpPrincipleClearcoatID, params, nullptr, nullptr);
    }
};

struct KpPrincipleBSSRDF {
    static RGBSpectrum eval(ShadingContext*);
    static RGBSpectrum sample(ShadingContext*, Sampler*);
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_FINISH_PARAM(KpPrincipleBSSRDFParams)
        };

        shadingsys.register_closure("kp_principle_bssrdf", KpPrincipleBSSRDFID, params, nullptr, nullptr);
    }
};

using bsdf_eval_func = std::function<RGBSpectrum(ShadingContext*)>;
using bsdf_sample_func = std::function<RGBSpectrum(ShadingContext*, Sampler*)>;

// cpp 17 inlined constexpr variables will have external linkage and
// have only one copy among all included files
inline bsdf_eval_func get_bsdf_eval_func(ClosureID id) {
    static std::array<bsdf_eval_func, 29> eval_functions {
        Diffuse::eval,
        Phong::eval,
        nullptr,
        Ward::eval,
        Reflection::eval,
        Refraction::eval,
        nullptr,
        nullptr,
        //Microfacet::eval,
        nullptr,
        nullptr,
        Emission::eval,
        Background::eval,
        nullptr,
        nullptr,
        KpMirror::eval,
        KpDielectric::eval,
        KpMicrofacet::eval,
        KpEmitter::eval,
        KpGloss::eval,
        KpGlass::eval,
        KpPrincipleDiffuse::eval,
        KpPrincipleRetro::eval,
        KpPrincipleFakeSS::eval,
        KpPrincipleSheen::eval,
        KpPrincipleSpecularReflection::eval,
        nullptr,
        KpPrincipleClearcoat::eval,
        nullptr,
        nullptr
    };
    return eval_functions[id];
}

inline bsdf_sample_func get_bsdf_sample_func(ClosureID id) {
    static std::array<bsdf_sample_func, 29> sample_functions {
        Diffuse::sample,
        Phong::sample,
        nullptr,
        Ward::sample,
        Reflection::sample,
        Refraction::sample,
        nullptr,
        nullptr,
        //Microfacet::sample,
        nullptr,
        nullptr,
        Emission::sample,
        Background::sample,
        nullptr,
        nullptr,
        KpMirror::sample,
        KpDielectric::sample,
        KpMicrofacet::sample,
        KpEmitter::sample,
        KpGloss::sample,
        KpGlass::sample,
        KpPrincipleDiffuse::sample,
        KpPrincipleRetro::sample,
        KpPrincipleFakeSS::sample,
        KpPrincipleSheen::sample,
        KpPrincipleSpecularReflection::sample,
        nullptr,
        KpPrincipleClearcoat::sample,
        nullptr,
        nullptr
    };
    return sample_functions[id];
}