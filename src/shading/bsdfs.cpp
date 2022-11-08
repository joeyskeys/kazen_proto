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

RGBSpectrum Diffuse::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    auto params = reinterpret_cast<const DiffuseParams*>(data);
    auto n = base::to_vec3(params->N);
    sample.pdf = std::max(base::dot(sample.wo, n), 0.f) * constants::one_div_pi<float>();
    return sample.pdf;
}

RGBSpectrum Diffuse::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Diffuse;
    sample.wo = sample_hemisphere(Vec2f(rand[0], rand[1]));
    return eval(data, sg, sample);
}

RGBSpectrum Phong::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
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

RGBSpectrum Phong::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
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

RGBSpectrum OrenNayar::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    return 0;
}

RGBSpectrum OrenNayar::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    return 0;
}

RGBSpectrum Ward::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
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

RGBSpectrum Ward::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
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

RGBSpectrum Reflection::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.pdf = 0;
    return 0;
}

RGBSpectrum Reflection::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Specular;
    auto params = reinterpret_cast<const ReflectionParams*>(data);
    auto n = base::to_vec3(sg.N);
    auto i = -base::to_vec3(sg.I);
    auto cos_theta_i = base::dot(n, i);
    if (cos_theta_i > 0) {
        sample.wo = reflect(i, n);
        sample.pdf = 1.f;
        //return fresnel(cos_theta_i, params->eta, 1.f);
        return 1.f;
    }
    
    sample.pdf = 0.f;
    return 0.f;
}

RGBSpectrum Refraction::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    // mode shouldn't be set in eval function since one-sample MIS applied
    // in composite closure
    // sample.mode = ScatteringMode::Specular;
    sample.pdf = 0;
    return 0;
}

RGBSpectrum Refraction::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Specular;
    auto params = reinterpret_cast<const RefractionParams*>(data);
    auto n = base::to_vec3(sg.N);
    auto i = -base::to_vec3(sg.I);
    auto cos_theta_i = base::dot(n, i);
    auto eta = params->eta;

    if (cos_theta_i < 0.f) {
        eta = 1. / eta;
        n = -n;
    }
    sample.wo = refract(i, n, eta);

    if (base::is_zero(sample.wo)) {
        sample.pdf = 0.f;
        return 0.f;
    }

    sample.pdf = 1.f;
    return 1.f;
}

RGBSpectrum Transparent::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.pdf = 1.f;
    return 1.f;
}

RGBSpectrum Transparent::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Specular;
    sample.wo = base::to_vec3(sg.I);
    sample.pdf = 1.f;
    return 1.f;
}

RGBSpectrum Translucent::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.pdf = std::max(base::dot(sample.wo, base::to_vec3(sg.N)), 0.f) * constants::one_div_pi<float>();
    return constants::one_div_pi<float>();
}

RGBSpectrum Translucent::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Diffuse;
    sample.wo = -sample_hemisphere(base::head<2>(rand));
    sample.pdf = std::max(base::dot(sample.wo, base::to_vec3(sg.N)), 0.f) * constants::one_div_pi<float>();
    return constants::one_div_pi<float>();
}

RGBSpectrum Emission::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.pdf = std::max(base::dot(sample.wo, base::to_vec3(sg.N)), 0.f) * constants::one_div_pi<float>();
    return 1.f;
}

RGBSpectrum Emission::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Diffuse;
    sample.wo = sample_hemisphere();
    sample.pdf = std::max(base::dot(sample.wo, base::to_vec3(sg.N)), 0.f) * constants::one_div_pi<float>();
    return 1.f;
}

RGBSpectrum Background::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    return 0.f;
}

RGBSpectrum Background::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    return 0.f;
}

RGBSpectrum KpMirror::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.pdf = 0.f;
    return 0.f;
}

RGBSpectrum KpMirror::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Specular;

    sample.wo = reflect(base::to_vec3(-sg.I), base::to_vec3(sg.N));
    sample.pdf = 1.f;
    return 1.f;
}

RGBSpectrum KpDielectric::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    //sample.mode = ScatteringMode::Specular;
    sample.pdf = 0.f;
    return 0.f;
}

RGBSpectrum KpDielectric::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Specular;
    sample.pdf = 1.f;
    auto params = reinterpret_cast<const KpDielectricParams*>(data);
    auto i = -base::to_vec3(sg.I);
    auto n = base::to_vec3(sg.N);
    auto cos_theta_i = base::dot(i, n);
    auto f = fresnel(cos_theta_i, params->ext_ior, params->int_ior);

    auto sp = random3f();
    if (sp.x() < f) {
        sample.wo = reflect(i, n);
    }
    else {
        auto fac = params->int_ior / params->ext_ior;
        if (cos_theta_i < 0.f) {
            fac = params->ext_ior / params->int_ior;
            n = -n;
        }

        sample.wo = refract(-i, n, fac);
    }
    return 1.f;
}

RGBSpectrum KpMicrofacet::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    auto params = reinterpret_cast<const KpMicrofacetParams*>(data);
    auto wi = base::to_vec3(-sg.I);
    auto n = base::to_vec3(sg.N);
    auto cos_theta_i = base::dot(wi, n);
    auto cos_theta_o = base::dot(sample.wo, n);
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

RGBSpectrum KpMicrofacet::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
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

RGBSpectrum KpEmitter::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.pdf = 1;
    auto params = reinterpret_cast<const KpEmitterParams*>(data);
    return params->albedo;
}

RGBSpectrum KpEmitter::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Diffuse;
    sample.pdf = 1;
    auto params = reinterpret_cast<const KpEmitterParams*>(data);
    return params->albedo;
}

static const OSL::ustring u_ggx("ggx");
static const OSL::ustring u_beckmann("beckmann");
static const OSL::ustring u_default("default");

RGBSpectrum KpGloss::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
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
        auto mdf = MicrofacetInterface<BeckmannDist>(wi, params->xalpha, params->yalpha);
        D = mdf.D(m);
        G = mdf.G(sample.wo);
        F = fresnel_refl_dielectric(params->eta, base::dot(m, wi));
        sample.pdf = mdf.pdf(m);
    }
    else {
        auto mdf = MicrofacetInterface<GGXDist>(wi, params->xalpha, params->yalpha);
        D = mdf.D(m);
        G = mdf.G(sample.wo);
        F = fresnel_refl_dielectric(params->eta, std::abs(base::dot(m, wi)));
        sample.pdf = mdf.pdf(m);
    }

    return D * G * F / (4.f * cos_theta_i * cos_theta_o);
}

RGBSpectrum KpGloss::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    // Should we set the mode according to the roughness value?
    sample.mode = ScatteringMode::Diffuse;
    auto params = reinterpret_cast<const MicrofacetParams*>(data);
    auto wi = base::to_vec3(-sg.I);
    auto cos_theta_i = cos_theta(wi);
    if (cos_theta_i <= 0)
        return 0.f;

    float D, G, F;
    if (params->dist == u_beckmann) {
        auto mdf = MicrofacetInterface<BeckmannDist>(wi, params->xalpha, params->yalpha);
        const Vec3f m = mdf.sample_m(rand);
        sample.wo = reflect(wi, m);
        auto cos_theta_o = cos_theta(sample.wo);
        if (cos_theta_o == 0.f)
            return 0.f;

        const float D = mdf.D(m);
        const float G = mdf.G(sample.wo);
        const float F = fresnel_refl_dielectric(params->eta, std::abs(base::dot(m, wi)));

        sample.pdf = mdf.pdf(m);
    }
    else {
        auto mdf = MicrofacetInterface<GGXDist>(wi, params->xalpha, params->yalpha);
        const Vec3f m = mdf.sample_m(rand);
        sample.wo = reflect(wi, m);
        auto cos_theta_o = cos_theta(sample.wo);
        if (cos_theta_o == 0.f)
            return 0.f;

        const float D = mdf.D(m);
        const float G = mdf.G(sample.wo);
        const float F = fresnel_refl_dielectric(params->eta, std::abs(base::dot(m, wi)));

        sample.pdf = mdf.pdf(m);
    }

    return D * G * F / (4.f * cos_theta_i * cos_theta(sample.wo));
}

RGBSpectrum KpGlass::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
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
            auto mdf = MicrofacetInterface<BeckmannDist>(wi, params->xalpha, params->yalpha);
            sample.pdf = reflection_pdf(mdf, m, cos_mi);
            return eval_reflection(mdf, sample.wo, m, F);
        }
        else {
            auto mdf = MicrofacetInterface<GGXDist>(wi, params->xalpha, params->yalpha);
            sample.pdf = reflection_pdf(mdf, m, cos_mi);
            return eval_reflection(mdf, sample.wo, m, F);
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
            auto mdf = MicrofacetInterface<BeckmannDist>(wi, params->xalpha, params->yalpha);
            sample.pdf = refraction_pdf(mdf, sample.wo, m, eta);
            return eval_refraction(mdf, eta, sample.wo, m, 1.f - F);
        }
        else {
            auto mdf = MicrofacetInterface<GGXDist>(wi, params->xalpha, params->yalpha);
            sample.pdf = refraction_pdf(mdf, sample.wo, m, eta);
            return eval_refraction(mdf, eta, sample.wo, m, 1.f - F);
        }
    }
}

RGBSpectrum KpGlass::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
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
    auto mdf_b = MicrofacetInterface<BeckmannDist>(wi, params->xalpha,
        params->yalpha);
    auto mdf_g = MicrofacetInterface<GGXDist>(wi, params->xalpha,
        params->yalpha);
    if (params->dist == u_beckmann)
        m = mdf_b.sample_m(rand);
    else
        m = mdf_g.sample_m(rand);

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
            sample.pdf = F * reflection_pdf(mdf_b, m, cos_mi);
            return eval_reflection(mdf_b, sample.wo, m, F);
        }
        else {
            sample.pdf = F * reflection_pdf(mdf_g, m, cos_mi);
            return eval_reflection(mdf_g, sample.wo, m, F);
        }
    }
    else {
        // Refraction
        // TODO : compute refraction with the result of previous calculation
        sample.wo = refract(wi, m, eta);
        if (wi[1] * sample.wo[1] > 0.f)
            return 0.f;

        if (params->dist == u_beckmann) {
            sample.pdf = (1.f - F) * refraction_pdf(mdf_b, sample.wo, m, eta);
            return eval_refraction(mdf_b, eta, sample.wo, m, 1.f - F);
        }
        else {
            sample.pdf = (1.f - F) * refraction_pdf(mdf_g, sample.wo, m, eta);
            return eval_refraction(mdf_g, eta, sample.wo, m, 1.f - F);
        }
    }
}

/*
 * Calculate cosine values in shading space is convenient but when bumped normal comes
 * into play things are different.
 * We can actually remap the wi&wo in the OSL code, but need to do some algebra there.
 * Currently we strict to the easy way which simply calculate the cosine value with dot.
 */

RGBSpectrum KpPrincipleDiffuse::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    auto params = reinterpret_cast<const DiffuseParams*>(data);
    auto wi = base::to_vec3(-sg.I);
    auto n = base::to_vec3(sg.N);
    auto cos_theta_o = base::dot(n, sample.wo);
    auto cos_theta_i = base::dot(n, wi);
    if (cos_theta_o <= 0 || cos_theta_i <= 0)
        return 0;

    sample.pdf = std::max(cos_theta_o, 0.f) * constants::one_div_pi<float>();
    return constants::one_div_pi<float>() * 
        (1.f - 0.5f * pow(1 - cos_theta(wi), 5));
        (1.f - 0.5f * pow(1 - cos_theta_o, 5));
}

RGBSpectrum KpPrincipleDiffuse::sample(const void* data, const OSL::ShaderGlobals&sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Diffuse;
    sample.wo = sample_hemisphere(base::head<2>(rand));
    return eval(data, sg, sample);
}

RGBSpectrum KpPrincipleRetro::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    auto params = reinterpret_cast<const KpPrincipleRetroParams*>(data);
    auto wi = base::to_vec3(-sg.I);
    auto n = base::to_vec3(sg.N);
    auto cos_theta_o = base::dot(n, sample.wo);
    auto cos_theta_i = base::dot(n, wi);
    if (cos_theta_o <= 0 || cos_theta_i <= 0)
        return 0;

    sample.pdf = std::max(cos_theta_o, 0.f) * constants::one_div_pi<float>();
    // Checkout "Extending the Disney BRDF to a BSDF with Integrated Subsurface
    // Scattering" page 6 equation 4
    // https://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf
    // Here we follow the same convension used in the paper with Latex syntax
    auto F_L = pow(1. - cos_theta_o, 5.);
    auto F_V = pow(1. - cos_theta_i, 5.);
    auto wh = base::normalize(wi + sample.wo);
    auto R_R = 2.f * params->roughness * square(base::dot(sample.wo, wh));
    return constants::one_div_pi<float>() *
        R_R * (F_L + F_V + F_L * F_V * (R_R - 1.));
}

RGBSpectrum KpPrincipleRetro::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Diffuse;
    sample.wo = sample_hemisphere(base::head<2>(rand));
    return eval(data, sg, sample);
}

RGBSpectrum KpPrincipleFakeSS::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    // This part of code is copied from PBRT-v3
    // Same code signature is found in WADS's brdf explorer code
    // TODO : read Hanrahan's paper in detail
    auto params = reinterpret_cast<const KpPrincipleFakeSSParams*>(data);
    auto wi = base::to_vec3(-sg.I);
    auto wh = base::normalize(wi + sample.wo);
    auto cos_theta_h = base::dot(wh, sample.wo);

    // fss90 used to "flatten" retroreflection based on roughness
    float fss90 = square(cos_theta_h) * params->roughness;
    auto schlick_weight = [](auto cos_theta_v) {
        return std::pow(1. - cos_theta_v, 5.);
    };
    auto n = base::to_vec3(sg.N);
    auto abs_cos_theta_o = std::abs(base::dot(n, sample.wo));
    auto abs_cos_theta_i = std::abs(base::dot(n, wi));
    float fi = schlick_weight(abs_cos_theta_i);
    float fo = schlick_weight(abs_cos_theta_o);
    auto fss = std::lerp(1.f, fss90, fi) * std::lerp(1.f, fss90, fo);
    // 1.25 scale is used to (roughly) preserve albedo
    auto ss = 1.25f * (fss * (1. / (abs_cos_theta_i + abs_cos_theta_o) - .5f) + .5f);
    sample.pdf = std::max(cos_theta(sample.wo), 0.f) * constants::one_div_pi<float>();

    return ss * constants::one_div_pi<float>();
}

RGBSpectrum KpPrincipleFakeSS::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Diffuse;
    sample.wo = sample_hemisphere(base::head<2>(rand));
    return eval(data, sg, sample);
}

RGBSpectrum KpPrincipleSheen::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    auto params = reinterpret_cast<const KpPrincipleSheenParams*>(data);
    auto wi = base::to_vec3(-sg.I);
    auto n = base::to_vec3(sg.N);
    auto cos_theta_o = base::dot(n, sample.wo);
    auto cos_theta_i = base::dot(n, wi);
    if (cos_theta_o <= 0 || cos_theta_i <= 0)
        return 0;

    sample.pdf = std::max(cos_theta(sample.wo), 0.f) * constants::one_div_pi<float>();
    auto wh = base::normalize(wi + sample.wo);
    auto cos_theta_v = base::dot(wh, sample.wo);
    if (cos_theta_v < 1e-6)
        return params->sheen;
    else
        return params->sheen * pow(1. - cos_theta_v, 5.);
}

RGBSpectrum KpPrincipleSheen::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    sample.mode = ScatteringMode::Diffuse;
    sample.wo = sample_hemisphere(base::head<2>(rand));
    return eval(data, sg, sample);
}

static RGBSpectrum disney_fresnel_eval(
    const RGBSpectrum F0,
    const float cos_theta_o,
    const float eta,
    const float metallic)
{
    return base::lerp(RGBSpectrum(fresnel_dielectric(cos_theta_o, eta)),
        fresnel_schlick(F0, cos_theta_o), RGBSpectrum{metallic});
}

RGBSpectrum KpPrincipleSpecularReflection::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    auto params = reinterpret_cast<const KpPrincipleSpecularParams*>(data);
    auto wi = base::to_vec3(-sg.I);
    auto n = base::to_vec3(sg.N);
    auto cos_theta_o = base::dot(n, sample.wo);
    auto cos_theta_i = base::dot(n, wi);
    if (cos_theta_i * cos_theta_o < 0.)
        return 0.f;

    const Vec3f wh = base::normalize(wi + sample.wo);
    auto cos_om = base::dot(sample.wo, wh);
    if (cos_om == 0.f)
        return 0.f;

    auto mdf = MicrofacetInterface<GGXDist>(wi, params->xalpha, params->yalpha);
    // We have a problem here, the impl now will return value much larger than 1
    // which should never happen, look into it later
    auto D = std::min(4.f, mdf.D(wh));
    auto G = mdf.G(sample.wo);
    auto F = base::lerp(base::to_vec3(params->F0), RGBSpectrum{1},
        RGBSpectrum{fresnel_schlick(params->eta, cos_om)});
    //auto F = disney_fresnel_eval(params->F0, base::dot(sample.wo, wh),
        //params->eta, params->metallic);
    //auto F = base::lerp(base::to_vec3(params->F0), RGBSpectrum{1}, schlick_weight(cos_om));
    sample.pdf = std::max(0.f, mdf.pdf(wh));

    // A min function is used to limit the value in reasonable range
    return base::vec_min(RGBSpectrum{1.f}, D * G * F / (4.f * cos_theta_i * cos_theta_o));
    //return D * G * F / (4.f * cos_theta_i * cos_theta_o);
}

RGBSpectrum KpPrincipleSpecularReflection::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    // This is confusing... Consider changing it to Delta & NonDelta
    sample.mode = ScatteringMode::Diffuse;
    auto params = reinterpret_cast<const KpPrincipleSpecularParams*>(data);
    auto wi = base::to_vec3(-sg.I);
    auto mdf = MicrofacetInterface<GGXDist>(wi, params->xalpha, params->yalpha);
    const Vec3f wh = mdf.sample_m(rand);
    sample.wo = reflect(wi, wh);
    return eval(data, sg, sample);
}

// Following two functions are less general and just put it here
static inline float gtr1(const float alpha, const float cos_theta_v) {
    auto a2 = square(alpha);
    return (a2 - 1.) / (constants::pi<float>() * log(a2) * (1. + (a2 - 1.) *
        cos_theta_v));
}

static inline float clearcoat_g1(const float cos_theta_v) {
    // This is a simplified version of GGX G function with Smith profile and
    // fixed roughenss 0.25 (0.25 * 0.25 = 0.0625)
    // The G1 reduced to:
    //                2
    // --------------------------------
    // 1 + sqrt(1 + a^2 * tan^2(theta))
    //
    // PBRT used another form that I couldn't understand, need to look into it
    // later.

    const auto sin_theta_v = std::sqrt(std::max(0., 1. - square(cos_theta_v)));
    const auto tan_theta_v = sin_theta_v / cos_theta_v;
    return 2. / (1. + std::sqrt(1. + 0.0625 * tan_theta_v));
}

RGBSpectrum KpPrincipleClearcoat::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    auto params = reinterpret_cast<const KpPrincipleClearcoatParams*>(data);
    auto wi = base::to_vec3(-sg.I);
    auto n = base::to_vec3(sg.N);
    auto cos_theta_o = base::dot(n, sample.wo);
    auto cos_theta_i = base::dot(n, wi);
    if (cos_theta_i <= 0 || cos_theta_o <= 0)
        return 0.f;

    const Vec3f wh = base::normalize(wi + sample.wo);
    auto cos_theta_h = cos_theta(wh);
    auto D = gtr1(params->roughness, cos_theta_h);
    auto G = clearcoat_g1(cos_theta_i) * clearcoat_g1(cos_theta_o);
    // Fixed ior 1.5, corresponding F0 = 0.04, here we write the schlick function
    // directly with precomputed F0
    // Perhaps we can make it constexpr?
    auto F = 0.04 + 0.96 * pow(1. - cos_theta_i, 5.);

    // PDF is documented in the paper appendix B.1
    sample.pdf = std::max(0., D * cos_theta_h / (4. * base::dot(sample.wo, wh)));

    // Parameter range is normalized into range of [0, 0.25]
    return std::clamp(D * G * F * 0.25, 0., 0.25);
}

RGBSpectrum KpPrincipleClearcoat::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    auto params = reinterpret_cast<const KpPrincipleClearcoatParams*>(data);
    // Check "Physically Based Shading at Disney" appendix B page 25 equations 2, 5
    // for the sampling function for GTR1
    auto phi = constants::two_pi<float>() * rand[0];
    auto a2 = square(params->roughness);
    auto cos_theta_v = std::sqrt((1. - std::pow(a2, 1. - rand[1])) / (1. - a2));
    auto sin_theta_v = std::sqrt(1. - square(cos_theta_v));
    float sin_phi_v, cos_phi_v;
    sincosf(phi, &sin_phi_v, &cos_phi_v);
    Vec3f wh{sin_theta_v * cos_phi_v, cos_theta_v, sin_theta_v * sin_phi_v};
    auto wi = base::to_vec3(-sg.I);
    sample.wo = reflect(wi, wh);

    return eval(data, sg, sample);
}

RGBSpectrum KpPrincipleBSSRDF::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    // Check "Approximate Reflectance Profiles for Efficient Subsurface Scattering"
    // page 4 equation 5
    auto params = reinterpret_cast<const KpPrincipleBSSRDFParams*>(data);
    const auto r = sample.bssrdf_r;
    const auto d = params->scatter_distance[sample.bssrdf_idx];
    sample.pdf = (.25f * std::exp(-r / d) / (2. * constants::pi<float>() * d * r)) +
        .75f * std::exp(-r / (3. * d)) / (6. * constants::pi<float>() * d * r);
    auto sd = base::to_vec3(params->scatter_distance);
    /*
    return base::exp(-RGBSpectrum{sample.bssrdf_r} / sd) +
        base::exp(-RGBSpectrum{sample.bssrdf_r} / (3.f * sd)) /
        (8. * constants::pi<float>() * sample.bssrdf_r * sd);
    */
    // Understanding of BSSRDF is not enough, leave the mess here for now
    return 0;
}

RGBSpectrum KpPrincipleBSSRDF::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec3f& rand) {
    auto params = reinterpret_cast<const KpPrincipleBSSRDFParams*>(data);
    uint32_t idx = rand[0] * 2.999999;
    sample.bssrdf_idx = idx;

    // The cdf is 1 - e^(-x/d) / 4 - (3 / 4) e^(-x / (3d))
    // Follow the suggestion in "Approximate Reflectance Profiles for Efficient Subsurface Scattering"
    // here we "randomly pick one of the two exponents, use its inverse as a cdf, and then weight the
    // results using MIS", the implementation is copied from pbrt-v3
    // Constants like 2.999999 or 3.99999 are used to avoid zero value that might
    // cause problem
    // TODO : go through the equation deduction

    float u;
    if (rand[1] < .25f) {
        u = rand[1] * 3.999999;
        sample.bssrdf_r = params->scatter_distance[idx] * std::log(1. / (1. - u));
    }
    else {
        u = (rand[1] - .25f) / .751111f;
        sample.bssrdf_r = 3. * params->scatter_distance[idx] * std::log(1. / (1. - u));
    }

    // The outgoing direction of bssrdf is not defined in the paper,
    // Just use cosine weighted distribution for now.
    // This's only need for the diffusion profile approximation, for real path
    // traced BSSRDF wo is calculated in the volume integrator.
    sample.wo = sample_hemisphere(Vec2f{u, rand[2]});

    return eval(data, sg, sample);
}