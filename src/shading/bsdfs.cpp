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

float Diffuse::eval(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& wo, float& pdf) {
    auto params = reinterpret_cast<const DiffuseParams*>(data);
    pdf = std::max(dot(wo, static_cast<Vec3f>(params->N)), 0.f) * constants::one_div_pi<float>();
    return 1.f;
}

float Diffuse::sample(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wo, float& pdf) {
    auto params = reinterpret_cast<const DiffuseParams*>(data);
    wo = sample_hemisphere();
    wo = tangent_to_world(wo, sg.N);
    pdf = std::max(dot(wo, params->N), 0.f) * constants::one_div_pi<float>();
    return 1.f;
}

float Phong::eval(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& wo, float& pdf) {
    auto params = reinterpret_cast<const PhongParams*>(data);
    float cos_ni = dot(static_cast<Vec3f>(params->N), wo);
    float cos_no = dot(static_cast<Vec3f>(-params->N), sg.I);
    if (cos_ni > 0 && cos_no > 0) {
        Vec3f R = (2 * cos_no) * params->N + sg.I;
        float cos_ri = dot(R, wo);
        if (cos_ri > 0) {
            pdf = (params->exponent + 1) * constants::one_div_two_pi<float>()
                * std::pow(cos_ri, params->exponent);
            return cos_ni * (params->exponent + 2) / (params->exponent + 1);
        }
    }

    return pdf = 0;
}

float Phong::sample(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wo, float& pdf) {
    auto params = reinterpret_cast<const PhongParams*>(data);
    float cos_no = dot(static_cast<Vec3f>(-params->N), sg.I);
    if (cos_no > 0) {
        Vec3f R = (2 * cos_no) * params->N + sg.I;
        wo = sample_hemisphere_with_exponent(params->exponent);
        auto v_cos_theta = wo.y();
        wo = tangent_to_world(wo, R);
        float cos_ni = dot(static_cast<Vec3f>(params->N), wo);
        if (cos_ni > 0) {
            pdf = (params->exponent + 1) * constants::one_div_two_pi<float>()
                * std::pow(v_cos_theta, params->exponent);
            return cos_ni * (params->exponent + 2) / (params->exponent + 1);
        }
    }

    return pdf = 0;
}

float Reflection::eval(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& wo, float& pdf) {
    pdf = 0;
    return 0;
}

float Reflection::sample(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wo, float& pdf) {
    auto params = reinterpret_cast<const ReflectionParams*>(data);
    auto cos_theta_i = -dot(Vec3f(sg.N), Vec3f(sg.I));
    if (cos_theta_i > 0) {
        wo = reflect(sg.I, sg.N);
        pdf = 1.f;
        return fresnel(cos_theta_i, 1.f, params->eta);
    }
    
    pdf = 0.f;
    return 0.f;
}

float Emission::eval(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& wo, float& pdf) {
    pdf = std::max(dot(wo, sg.N), 0.f) * constants::one_div_pi<float>();
    return 1.f;
}

float Emission::sample(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wo, float& pdf) {
    wo = sample_hemisphere();
    wo = tangent_to_world(wo, sg.N, sg.dPdu, sg.dPdv);
    pdf = std::max(dot(wo, sg.N), 0.f) * constants::one_div_pi<float>();
    return 1.f;
}

float Mirror::eval(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& wo, float& pdf) {
    pdf = 0.f;
    return 0.f;
}

float Mirror::sample(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wo, float& pdf) {
    if (cos_theta(wo) <= 0)
        return 0.f;

    wo = reflect(sg.I);
    return 1.f;
}

float Dielectric::eval(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& wo, float& pdf) {
    pdf = 0.f;
    return 0.f;
}

float Dielectric::sample(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wo, float& pdf) {
    auto params = reinterpret_cast<const DielectricParams*>(data);
    auto cos_theta_i = cos_theta(sg.I);
    auto f = fresnel(cos_theta_i, params->ext_ior, params->int_ior);

    if (sample.x() < f) {
        wo = reflect(sg.I);
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