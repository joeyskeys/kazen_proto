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

struct Diffuse {
    static float sample(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wi, float& pdf) {
        auto params = reinterpret_cast<DiffuseParams>(data);
        wi = sample_hemisphere();
        wi = tangent_to_world(wi, sg.N, sg.dPdu, sg.dPdv);
        pdf = std::max(dot(wi, params.N), 0.f) * boost::math::constants::one_div_pi<float>();
        return 1.f;
    }

    static float eval(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf) {
        pdf = std::max(dot(wi, static_cast<Vec3f>(params.N)), 0.f) * boost::math::constants::one_div_pi<float>();
        return 1.f;
    }
};

struct Phong {

};

struct Emission {
    static float sample(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wi, float& pdf) {
        wi = sample_hemisphere();
        wi = tangent_to_world(wi, sg.N, sg.dPdu, sg.dPdv);
        pdf = std::max(dot(wi, sg.N), 0.f) * boost::math::constants::one_div_pi<float>();
        return 1.f;
    }

    static float eval(const void* data, const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf) {
        pdf = std::max(dot(wi, sg.N), 0.f) * boost::math::constants::one_div_pi<float>();
        return 1.f;
    }
};