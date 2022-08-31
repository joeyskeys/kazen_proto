#include <algorithm>

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

// TODO : ubiquitous to_vec3 call makes it obvious the OSL::Vec3 and Vec3 type
// conversion is everywhere in the shading related calculation, for sake of both
// code clarity and performance, a unified way of vector calculation must be done

float Diffuse::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    auto params = reinterpret_cast<const DiffuseParams*>(data);
    sample.pdf = std::max(cos_theta(sample.wo), 0.f) * constants::one_div_pi<float>();
    //return 1.f;
    return constants::one_div_pi<float>();
}

float Diffuse::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Diffuse;
    auto params = reinterpret_cast<const DiffuseParams*>(data);
    sample.wo = sample_hemisphere(Vec2f(rand[0], rand[1]));
    sample.pdf = std::max(cos_theta(sample.wo), 0.f) * constants::one_div_pi<float>();
    //return 1.f;
    return constants::one_div_pi<float>();
}

float Phong::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    auto params = reinterpret_cast<const PhongParams*>(data);
    float cos_ni = cos_theta(sample.wo);
    float cos_no = base::dot(base::to_vec3(-params->N), base::to_vec3(sg.I));
    if (cos_ni > 0 && cos_no > 0) {
        Vec3f R = base::to_vec3((2 * cos_no) * params->N + sg.I);
        float cos_ri = base::dot(R, sample.wo);
        if (cos_ri > 0) {
            sample.pdf = (params->exponent + 1) * constants::one_div_two_pi<float>()
                * std::pow(cos_ri, params->exponent);
            return cos_ni * (params->exponent + 2) / (params->exponent + 1);
        }
    }

    return sample.pdf = 0;
}

float Phong::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Diffuse;
    auto params = reinterpret_cast<const PhongParams*>(data);
    float cos_no = base::dot(base::to_vec3(-params->N), base::to_vec3(sg.I));
    if (cos_no > 0) {
        Vec3f R = base::to_vec3((2 * cos_no) * params->N + sg.I);
        sample.wo = sample_hemisphere_with_exponent(Vec2f(rand[0], rand[1]), params->exponent);
        auto v_cos_theta = sample.wo.y();
        sample.wo = local_to_world(sample.wo, R);
        float cos_ni = dot(base::to_vec3(params->N), sample.wo);
        if (cos_ni > 0) {
            sample.pdf = (params->exponent + 1) * constants::one_div_two_pi<float>()
                * std::pow(v_cos_theta, params->exponent);
            return cos_ni * (params->exponent + 2) / (params->exponent + 1);
        }
    }

    return sample.pdf = 0;
}

float OrenNayar::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {

}

float OrenNayar::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    
}

float Ward::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    auto params = reinterpret_cast<const WardParams*>(data);
    auto wi = base::to_vec3(-sg.I);
    auto cos_theta_i = cos_theta(wi);
    auto cos_theta_o = cos_theta(sample.wo);
    if (cos_theta_i <= 0 || cos_theta_o <= 0)
        return 0.f;

    auto m = base::normalize(wi + sample.wo);
    auto sin_theta_v = sin_theta(wi);
    auto A = stretched_roughness(m, sin_theta_v, params->xalpha, params->yalpha);
    auto tmp = std::exp(-tan_2_theta(m) * A) /
        (4.f * constants::pi<float>() * params->xalpha * params->yalpha);

    // Checkout "Notes on the Ward BRDF" page 1,2 equation 3, 9.
    // https://www.graphics.cornell.edu/~bjw/wardnotes.pdf
    sample.pdf = std::max(tmp / (base::dot(m, wi) * std::pow(cos_theta(m), 3)), 0.);
    return tmp / (std::sqrt(cos_theta_i * cos_theta_o));
}

float Ward::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Diffuse;
    auto params = reinterpret_cast<const WardParams*>(data);
    auto wi = base::to_vec3(-sg.I);
    auto cos_theta_i = cos_theta(wi);
    if (cos_theta_i <= 0)
        return 0.f;

    // Checkout "Notes on the Ward BRDF" page 2 equation 6, 7.
    float phi, theta, sin_phi_v, cos_phi_v, sin_theta_v, cos_theta_v, A;
    if (params->xalpha == params->yalpha) {
        theta = std::atan(params->xalpha * std::sqrt(-std::log(rand[0])));
        phi = constants::two_pi<float>() * rand[1];
        sincosf(theta, &sin_theta_v, &cos_theta_v);
        sincosf(phi, &sin_phi_v, &cos_phi_v);
    }
    else {
        float phi = std::atan(params->yalpha / params->xalpha * std::tan(constants::two_pi<float>() * rand[0]));
        sincosf(phi, &sin_phi_v, &cos_phi_v);
        A = square(cos_phi_v / params->xalpha) + square(sin_phi_v / params->yalpha);
        float theta = std::atan(std::sqrt(-std::log(rand[1]) / A));
        sincosf(theta, &sin_theta_v, &cos_theta_v);
    }

    Vec3f m{sin_theta_v * cos_phi_v, cos_theta_v, sin_theta_v * sin_phi_v};
    sample.wo = reflect(wi, m);
    auto tmp = std::exp(-tan_2_theta(m) * A) /
        (4.f * constants::pi<float>() * params->xalpha * params->yalpha);
    
    // Same as above
    sample.pdf = std::max(tmp / (base::dot(m, wi) * std::pow(cos_theta_v, 3)), 0.);
    return tmp / (std::sqrt(cos_theta_i * cos_theta(sample.wo)));
} 

float Reflection::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.pdf = 0;
    return 0;
}

float Reflection::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Specular;
    auto params = reinterpret_cast<const ReflectionParams*>(data);
    auto cos_theta_i = -cos_theta(base::to_vec3(sg.I));
    if (cos_theta_i > 0) {
        sample.wo = reflect(base::to_vec3(-sg.I), base::to_vec3(sg.N));
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

float Refraction::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Specular;
    auto params = reinterpret_cast<const RefractionParams*>(data);
    auto cos_theta_i = -cos_theta(base::to_vec3(sg.I));
    auto n = base::to_vec3(sg.N);
    auto eta = params->eta;

    if (cos_theta_i < 0.f) {
        eta = 1. / eta;
        n = -n;
    }
    sample.wo = refract(base::to_vec3(sg.I), n, eta);

    if (base::is_zero(sample.wo)) {
        sample.pdf = 0.f;
        return 0.f;
    }

    sample.pdf = 1.f;
    return 1.f;
}

float Transparent::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.pdf = 1.f;
    return 1.f;
}

float Transparent::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Specular;
    sample.wo = base::to_vec3(sg.I);
    sample.pdf = 1.f;
    return 1.f;
}

float Translucent::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.pdf = std::max(cos_theta(sample.wo), 0.f) * constants::one_div_pi<float>();
    return constants::one_div_pi<float>();
}

float Translucent::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Diffuse;
    sample.wo = -sample_hemisphere(base::head<2>(rand));
    sample.pdf = std::max(cos_theta(sample.wo), 0.f) * constants::one_div_pi<float>();
    return constants::one_div_pi<float>();
}

float Emission::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.pdf = std::max(cos_theta(sample.wo), 0.f) * constants::one_div_pi<float>();
    return 1.f;
}

float Emission::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Diffuse;
    sample.wo = sample_hemisphere();
    sample.pdf = std::max(cos_theta(sample.wo), 0.f) * constants::one_div_pi<float>();
    return 1.f;
}

float KpMirror::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.pdf = 0.f;
    return 0.f;
}

float KpMirror::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Specular;

    sample.wo = reflect(base::to_vec3(-sg.I));
    sample.pdf = 1.f;
    return 1.f;
}

float KpDielectric::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    //sample.mode = ScatteringMode::Specular;
    sample.pdf = 0.f;
    return 0.f;
}

float KpDielectric::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Specular;
    sample.pdf = 1.f;
    auto params = reinterpret_cast<const KpDielectricParams*>(data);
    auto cos_theta_i = cos_theta(base::to_vec3(-sg.I));
    auto f = fresnel(cos_theta_i, params->ext_ior, params->int_ior);

    auto sp = random3f();
    if (sp.x() < f) {
        sample.wo = reflect(base::to_vec3(-sg.I));
    }
    else {
        auto n = base::to_vec3(sg.N);
        auto fac = params->int_ior / params->ext_ior;
        if (cos_theta_i < 0.f) {
            fac = params->ext_ior / params->int_ior;
            n = -n;
        }

        sample.wo = refract(base::to_vec3(sg.I), n, fac);
    }
    return 1.f;
}

float KpMicrofacet::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    auto params = reinterpret_cast<const KpMicrofacetParams*>(data);
    auto wi = base::to_vec3(-sg.I);
    auto cos_theta_i = cos_theta(base::to_vec3(-sg.I));
    auto cos_theta_o = cos_theta(sample.wo);
    if (cos_theta_i <= 0 || cos_theta_o <= 0)
        return 0.f;

    auto wh = base::normalize(wi + sample.wo);
    auto D = BeckmannPDF(wh, params->alpha);
    auto F = fresnel(base::dot(wh, wi), params->ext_ior, params->int_ior);
    auto G = G1(wh, wi, params->alpha) * G1(wh, sample.wo, params->alpha);
    auto ks = 1. - params->kd;
    auto Jh = 1. / (4. * base::dot(wh, sample.wo));
    sample.pdf = ks * D * Jh + params->kd * cos_theta_o * constants::one_div_pi<float>();

    return constants::one_div_pi<float>() +
        ks * D * F * G / (4. * cos_theta_i * cos_theta_o * cos_theta(wh));
}

float KpMicrofacet::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Diffuse;
    auto params = reinterpret_cast<const KpMicrofacetParams*>(data);
    auto ks = 1. - params->kd;
    auto wi = base::to_vec3(-sg.I);

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

float KpEmitter::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Diffuse;
    sample.pdf = 1;
    auto params = reinterpret_cast<const KpEmitterParams*>(data);
    return params->albedo;
}

static const OSL::ustring u_ggx("ggx");
static const OSL::ustring u_beckmann("beckmann");
static const OSL::ustring u_default("default");

float KpGloss::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    auto params = reinterpret_cast<const MicrofacetParams*>(data);
    auto wi = base::to_vec3(-sg.I);
    auto cos_theta_i = cos_theta(wi);
    auto cos_theta_o = cos_theta(sample.wo);
    if (cos_theta_i <= 0 || cos_theta_o <= 0)
        return 0.f;

    //auto f = fresnel_refl_dielectric(data->eta, cos_theta_i);
    auto f = [&params](const float cos_theta_v) {
        fresnel_refl_dielectric(params->eta, cos_theta_v);
    };

    const Vec3f m = base::normalize(wi + sample.wo);
    auto cos_om = base::dot(sample.wo, m);
    if (cos_om == 0.f)
        return 0.f;

    float D, G, F;
    if (params->dist == u_beckmann) {
        D = MicrofacetInterface<BeckmannDist>::D(m, params->xalpha, params->yalpha);
        G = MicrofacetInterface<BeckmannDist>::G(wi, sample.wo, params->xalpha, params->yalpha);
        F = fresnel_refl_dielectric(params->eta, base::dot(m, wi));
        sample.pdf = MicrofacetInterface<BeckmannDist>::pdf(wi, m, params->xalpha, params->yalpha);
    }
    else {
        D = MicrofacetInterface<GGXDist>::D(m, params->xalpha, params->yalpha);
        G = MicrofacetInterface<GGXDist>::G(wi, sample.wo, params->xalpha, params->yalpha);
        F = fresnel_refl_dielectric(params->eta, std::abs(base::dot(m, wi)));
        sample.pdf = MicrofacetInterface<GGXDist>::pdf(wi, m, params->xalpha, params->yalpha);
    }

    return D * G * F / (4.f * cos_theta_i * cos_theta_o);
}

float KpGloss::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    // Should we set the mode according to the roughness value?
    sample.mode = ScatteringMode::Diffuse;
    auto params = reinterpret_cast<const MicrofacetParams*>(data);
    auto wi = base::to_vec3(-sg.I);
    auto cos_theta_i = cos_theta(wi);
    if (cos_theta_i <= 0)
        return 0.f;

    float D, G, F;
    if (params->dist == u_beckmann) {
        const Vec3f m = MicrofacetInterface<BeckmannDist>::sample_m(wi,
            params->xalpha, params->yalpha, rand);
        sample.wo = reflect(wi, m);
        auto cos_theta_o = cos_theta(sample.wo);
        if (cos_theta_o == 0.f)
            return 0.f;

        const float D = MicrofacetInterface<BeckmannDist>::D(m, params->xalpha,
            params->yalpha);
        const float G = MicrofacetInterface<BeckmannDist>::G(wi, sample.wo,
            params->xalpha, params->yalpha);
        const float F = fresnel_refl_dielectric(params->eta, std::abs(base::dot(m, wi)));

        sample.pdf = MicrofacetInterface<BeckmannDist>::pdf(wi, m, params->xalpha,
            params->yalpha);
    }
    else {
        const Vec3f m = MicrofacetInterface<GGXDist>::sample_m(wi,
            params->xalpha, params->yalpha, rand);
        sample.wo = reflect(wi, m);
        auto cos_theta_o = cos_theta(sample.wo);
        if (cos_theta_o == 0.f)
            return 0.f;

        const float D = MicrofacetInterface<GGXDist>::D(m, params->xalpha,
            params->yalpha);
        const float G = MicrofacetInterface<GGXDist>::G(wi, sample.wo,
            params->xalpha, params->yalpha);
        const float F = fresnel_refl_dielectric(params->eta, std::abs(base::dot(m, wi)));

        sample.pdf = MicrofacetInterface<GGXDist>::pdf(wi, m, params->xalpha,
            params->yalpha);
    }

    return D * G * F / (4.f * cos_theta_i * cos_theta(sample.wo));
}

float KpGlass::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    auto params = reinterpret_cast<const MicrofacetParams*>(data);
    auto wi = base::to_vec3(-sg.I);
    auto cos_theta_i = cos_theta(wi);
    auto cos_theta_o = cos_theta(sample.wo);
    auto eta = params->eta;
    if (eta == 1.f) {
        sample.pdf = 0.f;
        return 0.f;
    }
    if (cos_theta_i < 0)
        eta = 1.f / eta;
    
    if (cos_theta_i * cos_theta_o >= 0.f) {
        // Reflect
        const Vec3f m = base::normalize(wi + sample.wo);
        auto cos_mi = base::dot(wi, m);
        const float F = fresnel_refl_dielectric(eta, cos_mi);
        if (params->dist == u_beckmann) {
            sample.pdf = reflection_pdf<BeckmannDist>(wi, m, cos_mi, params->xalpha, params->yalpha);
            return eval_reflection<BeckmannDist>(wi, sample.wo, m, params->xalpha, params->yalpha, F);
        }
        else {
            sample.pdf = reflection_pdf<GGXDist>(wi, m, cos_mi, params->xalpha, params->yalpha);
            return eval_reflection<GGXDist>(wi, sample.wo, m, params->xalpha, params->yalpha, F);
        }
    }
    else {
        // Refract
        Vec3f m = base::normalize(wi + eta * sample.wo);
        if (m[1] < 0.f)
            m = -m;
        auto cos_mi = base::dot(wi, m);
        const float F = fresnel_refl_dielectric(eta, cos_mi);
        if (params->dist == u_beckmann) {
            sample.pdf = refraction_pdf<BeckmannDist>(wi, sample.wo, m, params->xalpha, params->yalpha,
                eta);
            return eval_refraction<BeckmannDist>(eta, wi, sample.wo, m, params->xalpha, params->yalpha, 1.f - F);
        }
        else {
            sample.pdf = refraction_pdf<GGXDist>(wi, sample.wo, m, params->xalpha, params->yalpha,
                eta);
            return eval_refraction<GGXDist>(eta, wi, sample.wo, m, params->xalpha, params->yalpha, 1.f - F);
        }
    }
}

float KpGlass::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    // TODO : sample as KpGloss
    sample.mode = ScatteringMode::Diffuse;
    auto params = reinterpret_cast<const MicrofacetParams*>(data);
    auto wi = base::to_vec3(-sg.I);
    auto cos_theta_i = cos_theta(wi);
    float eta = params->eta;
    if (eta == 1.f) {
        sample.pdf = 0.f;
        return 0.f;
    }
    if (cos_theta_i < 0.f)
        eta = 1.f / eta;

    Vec3f m;
    if (params->dist == u_beckmann)
        m = MicrofacetInterface<BeckmannDist>::sample_m(wi, params->xalpha,
            params->yalpha, rand);
    else
        m = MicrofacetInterface<GGXDist>::sample_m(wi, params->xalpha,
            params->yalpha, rand);

    auto cos_mi = std::clamp(base::dot(m, wi), -1.f, 1.f);
    // We need a extra fresnel function which calculates cos_theta_t
    //float cos_theta_t;
    auto F = fresnel_refl_dielectric(eta, cos_mi);
    
    // Sampling between reflection and refraction
    if (rand[2] < F) {
        // Reflection
        sample.wo = reflect(wi, m);
        if (wi[1] * sample.wo[1] <= 0.f)
            return 0.f;

        if (params->dist == u_beckmann) {
            sample.pdf = F * reflection_pdf<BeckmannDist>(wi, m, cos_mi, params->xalpha, params->yalpha);
            return eval_reflection<BeckmannDist>(wi, sample.wo, m, params->xalpha, params->yalpha, F);
        }
        else {
            sample.pdf = F * reflection_pdf<GGXDist>(wi, m, cos_mi, params->xalpha, params->yalpha);
            return eval_reflection<GGXDist>(wi, sample.wo, m, params->xalpha, params->yalpha, F);
        }
    }
    else {
        // Refraction
        // TODO : compute refraction with the result of previous calculation
        sample.wo = refract(wi, m, eta);
        if (wi[1] * sample.wo[1] > 0.f)
            return 0.f;

        if (params->dist == u_beckmann) {
            sample.pdf = (1.f - F) * refraction_pdf<BeckmannDist>(wi, sample.wo, m, params->xalpha,
                params->yalpha, eta);
            return eval_refraction<BeckmannDist>(eta, wi, sample.wo, m, params->xalpha, params->yalpha,
                1.f - F);
        }
        else {
            sample.pdf = (1.f - F) * refraction_pdf<GGXDist>(wi, sample.wo, m, params->xalpha,
                params->yalpha, eta);
            return eval_refraction<GGXDist>(eta, wi, sample.wo, m, params->xalpha, params->yalpha,
                1.f - F);
        }
    }
}