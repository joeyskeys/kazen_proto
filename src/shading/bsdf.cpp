#include <numbers>

#include "bsdf.h"
#include "base/sampling.h"

enum ClosureID {
    // Just add a few basic closures for test first

    // BSDF closures
    DiffuseID,

    // Microfacet closures
    GlossyID,

    // Emission closures
    EmissionID,

    NumClosureIDs
};

class Diffuse : public BSDF {
public:
    struct Params {
        OSL::Vec3   N;
    };

    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(Params, N),
            CLOSURE_FINISH_PARAM(Params)
        };

        shadingsys.register_closure("diffuse", DiffuseID, params, nullptr, nullptr);
    }

    float eval(const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf) const override {
        pdf = std::max(dot(wi, N), 0.f) * std::numbers::inv_pi_v;
        return 1.f;
    }

    float sample(const OSL::ShaderGlobals& sg, Vec3f& wi, float& pdf) const override {
        wi = sample_hemisphere();
        pdf = std::max(dot(wi, N), 0.f) * std::numbers::inv_pi_v;
        return 1.f;
    }

private:
    Vec3f N;
    Params params;
};

class Glossy : public BSDF {

};

class Emission : public BSDF {

};

void register_closures(OSL::ShadingSystem *shadingsys) {

}

void process_closure(const OSL::ClosureColor *Ci, bool light_only) {

}