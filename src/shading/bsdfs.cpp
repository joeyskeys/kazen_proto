#include <boost/math/constants/constants.hpp>

#include "bsdfs.h"
#include "base/utils.h"
#include "base/vec.h"
#include "core/sampling.h"

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
    sample.pdf = std::max(dot(sample.wo, static_cast<Vec3f>(params->N)), 0.f) * constants::one_div_pi<float>();
    return 1.f;
}

float Diffuse::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.mode = ScatteringMode::Diffuse;
    auto params = reinterpret_cast<const DiffuseParams*>(data);
    sample.wo = sample_hemisphere();
    sample.wo = tangent_to_world(sample.wo, sg.N);
    sample.pdf = std::max(dot(sample.wo, params->N), 0.f) * constants::one_div_pi<float>();
    return 1.f;
}

float Phong::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.mode = ScatteringMode::Diffuse;
    auto params = reinterpret_cast<const PhongParams*>(data);
    float cos_ni = dot(static_cast<Vec3f>(params->N), sample.wo);
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
        sample.wo = tangent_to_world(sample.wo, R);
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
    auto cos_theta_i = -dot(Vec3f(sg.N), Vec3f(sg.I));
    if (cos_theta_i > 0) {
        sample.wo = reflect(-sg.I, sg.N);
        sample.pdf = 1.f;
        return fresnel(cos_theta_i, 1.f, params->eta);
    }
    
    sample.pdf = 0.f;
    return 0.f;
}

float Emission::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.mode = ScatteringMode::Diffuse;
    sample.pdf = std::max(dot(sample.wo, sg.N), 0.f) * constants::one_div_pi<float>();
    return 1.f;
}

float Emission::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.mode = ScatteringMode::Diffuse;
    sample.wo = sample_hemisphere();
    sample.wo = tangent_to_world(sample.wo, sg.N, sg.dPdu, sg.dPdv);
    sample.pdf = std::max(dot(sample.wo, sg.N), 0.f) * constants::one_div_pi<float>();
    return 1.f;
}

float Mirror::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.mode = ScatteringMode::Specular;
    sample.pdf = 0.f;
    return 0.f;
}

float Mirror::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.mode = ScatteringMode::Specular;
    if (cos_theta(sample.wo) <= 0)
        return 0.f;

    sample.wo = reflect(sg.I);
    return 1.f;
}

float Dielectric::eval(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.mode = ScatteringMode::Specular;
    sample.pdf = 0.f;
    return 0.f;
}

float Dielectric::sample(const void* data, const OSL::ShaderGlobals& sg, BSDFSample& sample) {
    sample.mode = ScatteringMode::Specular;
    auto params = reinterpret_cast<const DielectricParams*>(data);
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