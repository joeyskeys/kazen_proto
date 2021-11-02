#include <numbers>

#include "bsdf.h"
#include "base/sampling.h"

enum ClosureID {
    // Just add a few basic closures for test first

    // BSDF closures
    DiffuseID,

    // Microfacet closures
    //GlossyID,

    // Emission closures
    //EmissionID,

    NumClosureIDs
};

struct DiffuseParams    { OSL::Vec3 N; };

class Diffuse : public BSDF {
public:
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(Params, N),
            CLOSURE_FINISH_PARAM(Params)
        };

        shadingsys.register_closure("diffuse", DiffuseID, params, nullptr, nullptr);
    }

    float eval(const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf) const override {
        pdf = std::max(dot(wi, params.N), 0.f) * std::numbers::inv_pi_v;
        return 1.f;
    }

    float sample(const OSL::ShaderGlobals& sg, Vec3f& wi, float& pdf) const override {
        wi = sample_hemisphere();
        pdf = std::max(dot(wi, params.N), 0.f) * std::numbers::inv_pi_v;
        return 1.f;
    }

public:
    DiffuseParams params;
};

/*
class Glossy : public BSDF {
public:
    struct Params {
        OSL::Vec3   N;
        OSL::Vec3   T;
        float       roughness;
        float       anisotropy;
        float       ior;
        float       energy_compensation;
        float       fresnel_weight;
    };

    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(Params, N),
            CLOSURE_VECTOR_PARAM(Params, T),
            CLOSURE_FLOAT_PARAM(Params, roughness),
            CLOSURE_FLOAT_PARAM(Params, anisotropy),
            CLOSURE_FLOAT_PARAM(Params, ior),
            CLOSURE_FLOAT_PARAM(Params, energy_compensation),
            CLOSURE_FLOAT_PARAM(Params, fresnel_weight),
            CLOSURE_FINISH_PARAM(Params)
        };

        shadingsys.register_closure("glossy", GlossyID, params, nullptr, nullptr);
    }

    float eval(const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf) const override {

    }

    float sample(const OSL::ShaderGlobals& sg, Vec3f& wi, float& pdf) const override {

    }
};
*/

/*
class Emission : public BSDF {

};
*/

namespace
{
    template <typename ClosureType>
    void register_closure(OSLShadingSystem& shadingsys) {
        ClosureType::register_closure(shadingsys);
    }
}

void register_closures(OSL::ShadingSystem *shadingsys) {
    register_closure<Diffuse>(shadingsys);
}

void process_closure(ShadingResult& ret, const OSL::ClosureColor *closure, const RGBSpectrum& w, bool light_only) {
    if (!closure)
        return;

    switch (closure->id) {
        case OSL::ClosureColor::MUL:
            auto cw = w * closure->as_mul()->weight;
            process_closure(ret, closure->as_mul()->closure, cw, light_only);
            break;

        case OSL::ClosureColor::ADD:
            process_closure(ret, closure->as_add()->closureA, w, light_only);
            process_closure(ret, closure->as_add()->closureB, w, light_only);
            break;

        default:
            const ClosureComponent *comp = closure->as_comp();
            RGBSpectrum cw = w * comp->w;
            
            if (!light_only) {
                bool status = false;
                switch (comp-id) {
                    case DIFFUSE_ID:        status = ret.bsdf.add_bsdf<Diffuse, DiffuseParams>(cw, *comp->as<Diffuse>().params);
                        break;
                }
                OSL_ASSERT(status && "Invalid closure invoked");
            }

            break;
    }
}