#include <boost/math/constants/constants.hpp>

#include "bsdfs.h"
#include "base/utils.h"
#include "base/vec.h"
#include "core/sampling.h"
#include "shading/microfacet.h"

namespace constants = boost::math::constants;

/*
 * The design of seperating actual closure function and the OSL closure interface
 * have the following CONS:
 * 
 * 1. Closures contain emissive ones that will also be sampled by light at some-
 *    where else, seperate the actually closure computation code to let light
 *    reuse it;
 * 2. Avoid slow vitual function.
 */

float Diffuse::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.mode = ScatteringMode::Diffuse;
    auto params = reinterpret_cast<const DiffuseParams*>(data);
    sample.pdf = std::max(cos_theta(sample.wo), 0.f) * constants::one_div_pi<float>();
    return 1.f;
}

float Diffuse::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.mode = ScatteringMode::Diffuse;
    auto params = reinterpret_cast<const DiffuseParams*>(data);
    sample.wo = sample_hemisphere();
    sample.pdf = std::max(cos_theta(sample.wo), 0.f) * constants::one_div_pi<float>();
    return 1.f;
}

float Phong::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.mode = ScatteringMode::Diffuse;
    auto params = reinterpret_cast<const PhongParams*>(data);
    float cos_ni = cos_theta(sample.wo);
    float cos_no = dot(static_cast<Vec3f>(-params->N), sg.I);
    if (cos_ni > 0 && cos_no > 0) {
        Vec3f R = (2 * cos_no) * params->N + sg.I;
        float cos_ri = dot(R, sample.wo);
        if (cos_ri > 0) {
            sample.pdf = (params->exponent + 1) * constants::one_div_two_pi<float>()
                * std::pow(cos_ri, params->exponent);
            return cos_ni * (params->exponent + 2) / (params->exponent + 1);
        }
    }

    return sample.pdf = 0;
}

float Phong::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.mode = ScatteringMode::Diffuse;
    auto params = reinterpret_cast<const PhongParams*>(data);
    float cos_no = dot(static_cast<Vec3f>(-params->N), sg.I);
    if (cos_no > 0) {
        Vec3f R = (2 * cos_no) * params->N + sg.I;
        sample.wo = sample_hemisphere_with_exponent(params->exponent);
        auto v_cos_theta = sample.wo.y();
        sample.wo = local_to_world(sample.wo, R);
        float cos_ni = dot(static_cast<Vec3f>(params->N), sample.wo);
        if (cos_ni > 0) {
            sample.pdf = (params->exponent + 1) * constants::one_div_two_pi<float>()
                * std::pow(v_cos_theta, params->exponent);
            return cos_ni * (params->exponent + 2) / (params->exponent + 1);
        }
    }

    return sample.pdf = 0;
}

float Reflection::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.mode = ScatteringMode::Specular;
    sample.pdf = 0;
    return 0;
}

float Reflection::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.mode = ScatteringMode::Specular;
    auto params = reinterpret_cast<const ReflectionParams*>(data);
    auto cos_theta_i = -cos_theta(sg.I);
    if (cos_theta_i > 0) {
        sample.wo = reflect(-sg.I, sg.N);
        sample.pdf = 1.f;
        //return fresnel(cos_theta_i, params->eta, 1.f);
        return 1.f;
    }
    
    sample.pdf = 0.f;
    return 0.f;
}

float Refraction::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    // mode shouldn't be set in eval function since one-sample MIS applied
    // in composite closure
    // sample.mode = ScatteringMode::Specular;
    sample.pdf = 0;
    return 0;
}

float Refraction::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.mode = ScatteringMode::Specular;
    auto params = reinterpret_cast<const RefractionParams*>(data);
    auto cos_theta_i = -cos_theta(Vec3f(sg.I));
    auto n = sg.N;
    auto eta = params->eta;

    if (cos_theta_i < 0.f) {
        eta = 1. / eta;
        n = -n;
    }
    sample.wo = refract(sg.I, n, eta);

    if (sample.wo.is_zero()) {
        sample.pdf = 0.f;
        return 0.f;
    }

    sample.pdf = 1.f;
    return 1.f;
}

float Microfacet::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    auto params = reinterpret_cast<const MicrofacetAnisoParams*>(data);
    auto wo = sample.wo;
    auto wi = -sg.I;

    auto cos_theta_i = cos_theta(wi);
    auto cos_theta_o = cos_theta(wo);
    if (cos_theta_i <= 0 || cos_theta_o)
        return 0;

    auto wh = (wo + wi).normalized();
    auto D = BeckmannPDF(wh, params->xalpha);
    //auto Jh = 1. / (4. * dot(wh, wo));
    auto F = fresnel(dot(wh, wi), 1, params->eta);
    auto G = G1(wh, wi, params->xalpha) * G1(wh, wo, params->xalpha);
    //sample.pdf = D * Jh;
    sample.pdf = D;

    return D * F * G / (4. * cos_theta_i * cos_theta_o * cos_theta(wh));
}

float Microfacet::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.mode = ScatteringMode::Diffuse;
    auto params = reinterpret_cast<const MicrofacetParams*>(data);
    auto wh = BeckmannMDF(random2f(), params->alpha);
    sample.wo = reflect(-sg.I, wh);
    auto f = eval(data, sg, sample);
    return f;
}

float MicrofacetAniso::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    auto params = reinterpret_cast<const MicrofacetAnisoParams*>(data);
    auto wo = sample.wo;
    auto wi = -sg.I;

    auto cos_theta_i = cos_theta(wi);
    auto cos_theta_o = cos_theta(wo);
    if (cos_theta_i <= 0 || cos_theta_o <= 0)
        return 0;

    auto wh = (wo + wi).normalized();
    auto D = BeckmannPDF(wh, params->xalpha);
    //auto Jh = 1. / (4. * dot(wh, wo));
    auto F = fresnel(dot(wh, wi), 1, params->eta);
    auto G = G1(wh, wi, params->xalpha) * G1(wh, wo, params->xalpha);
    //sample.pdf = D * Jh;
    sample.pdf = D;

    return D * F * G / (4. * cos_theta_i * cos_theta_o * cos_theta(wh));
}

float MicrofacetAniso::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.mode = ScatteringMode::Diffuse;
    auto params = reinterpret_cast<const MicrofacetAnisoParams*>(data);
    auto wh = BeckmannMDF(random2f(), params->xalpha);
    sample.wo = reflect(-sg.I, wh);
    auto f = eval(data, sg, sample);
    return f;
}

float Emission::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.mode = ScatteringMode::Diffuse;
    sample.pdf = std::max(cos_theta(sample.wo), 0.f) * constants::one_div_pi<float>();
    return 1.f;
}

float Emission::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.mode = ScatteringMode::Diffuse;
    sample.wo = sample_hemisphere();
    sample.pdf = std::max(cos_theta(sample.wo), 0.f) * constants::one_div_pi<float>();
    return 1.f;
}

float KpMirror::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.pdf = 0.f;
    return 0.f;
}

float KpMirror::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.mode = ScatteringMode::Specular;

    sample.wo = reflect(-sg.I);
    sample.pdf = 1.f;
    return 1.f;
}

float KpDielectric::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.mode = ScatteringMode::Specular;
    sample.pdf = 0.f;
    return 0.f;
}

float KpDielectric::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.mode = ScatteringMode::Specular;
    sample.pdf = 1.f;
    auto params = reinterpret_cast<const KpDielectricParams*>(data);
    auto cos_theta_i = cos_theta(sg.I);
    auto f = fresnel(cos_theta_i, params->ext_ior, params->int_ior);

    auto sp = random3f();
    if (sp.x() < f) {
        sample.wo = reflect(sg.I);
    }
    else {
        auto n = Vec3f(0.f, 1.f, 0.f);
        auto fac = params->int_ior / params->ext_ior;
        if (cos_theta_i < 0.f) {
            fac = params->ext_ior / params->int_ior;
            n[1] = -1.f;
        }

        refract(-sg.I, n, fac);
    }
    return 1.f;
}

float KpMicrofacet::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    auto params = reinterpret_cast<const KpMicrofacetParams*>(data);
    auto wi = Vec3f(-sg.I);
    auto cos_theta_i = cos_theta(-sg.I);
    auto cos_theta_o = cos_theta(sample.wo);
    if (cos_theta_i <= 0 || cos_theta_o <= 0)
        return 0.f;

    auto wh = (wi + sample.wo).normalized();
    auto D = BeckmannPDF(wh, params->alpha);
    auto F = fresnel(dot(wh, wi), params->ext_ior, params->int_ior);
    auto G = G1(wh, wi, params->alpha) * G1(wh, sample.wo, params->alpha);
    auto ks = 1. - params->kd;
    auto Jh = 1. / (4. * dot(wh, sample.wo));
    sample.pdf = ks * D * Jh + params->kd * cos_theta_o * constants::one_div_pi<float>();

    return constants::one_div_pi<float>() +
        ks * D * F * G / (4. * cos_theta_i * cos_theta_o * cos_theta(wh));
}

float KpMicrofacet::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    auto params = reinterpret_cast<const KpMicrofacetParams*>(data);
    auto ks = 1. - params->kd;
    auto wi = Vec3f(-sg.I);

    if (randomf() < ks) {
        auto n = BeckmannMDF(random2f(), params->alpha);
        sample.wo = reflect(wi, n);
    }
    else {
        sample.wo = to_cosine_hemisphere(random2f());
    }

    return eval(data, sg, sample) * cos_theta(sample.wo);
}

float KpEmitter::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.pdf = 1;
    auto params = reinterpret_cast<const KpEmitterParams*>(data);
    return params->albedo;
}

float KpEmitter::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.mode = ScatteringMode::Diffuse;
    sample.pdf = 1;
    auto params = reinterpret_cast<const KpEmitterParams*>(data);
    return params->albedo;
}