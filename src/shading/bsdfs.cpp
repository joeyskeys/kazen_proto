#include "bsdfs.h"
#include "base/vec.h"

/*
 * The design of seperating actual closure function and the OSL closure interface
 * have the following CONS:
 * 
 * 1. Closures contain emissive ones that will also be sampled by light at some-
 *    where else, seperate the actually closure computation code to let light
 *    reuse it;
 * 2. Avoid slow vitual function.
 */

float Diffuse::eval(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf) {
    pdf = std::max(dot(wi, static_cast<Vec3f>(params.N)), 0.f) * boost::math::constants::one_div_pi<float>();
    return 1.f;
}

float Diffuse::sample(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wi, float& pdf) {
    auto params = reinterpret_cast<DiffuseParams*>(data);
    wi = sample_hemisphere();
    wi = tangent_to_world(wi, sg.N, sg.dPdu, sg.dPdv);
    pdf = std::max(dot(wi, params->N), 0.f) * boost::math::constants::one_div_pi<float>();
    return 1.f;
}

float Phong::eval(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf) {
    auto params = reinterpret_cast<PhongParams*>(data);
    float cos_ni = dot(params->N, wi);
    float cos_no = dot(-params->N, sg.I);
    if (cos_ni > 0 && cos_no > 0) {
        auto R = (2 * cos_no) * params->N + sg.I;
        float cos_ri = dot(R, wi);
        if (cos_ri > 0) {
            pdf = (exponent + 1) * boost::math::constants::one_div_two_pi<float>()
                * std::pow(cos_ri, exponent);
            return cos_ni * (exponent + 2) / (exponent + 1);
        }
    }

    return pdf = 0;
}

float Phong::sample(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wi, float& pdf) {
    auto params = reinterpret_cast<PhongParams*>(data);
    float cos_no = dot(-params->N, sg.I);
    if (cos_no > 0) {
        Vec3f R = (2 * cos_no) * params->N + sg.I;
        wi = sample_hemisphere_with_exponent(exponent);
        auto v_cos_theta = wi.y();
        wi = tangent_to_world(wi, R);
        float cos_ni = dot(params->N, wi);
        if (cos_ni > 0) {
            pdf = (exponent + 1) * boost::math::constants::one_div_two_pi<float>()
                * std::pow(v_cos_theta, exponent);
            return cos_ni * (exponent + 2) / (exponent + 1);
        }
    }

    return pdf = 0;
}

float Emission::eval(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf) {
    pdf = std::max(dot(wi, sg.N), 0.f) * boost::math::constants::one_div_pi<float>();
    return 1.f;
}

float Emission::sample(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wi, float& pdf) {
    wi = sample_hemisphere();
    wi = tangent_to_world(wi, sg.N, sg.dPdu, sg.dPdv);
    pdf = std::max(dot(wi, sg.N), 0.f) * boost::math::constants::one_div_pi<float>();
    return 1.f;
}