// There's a removed feature of cpp20 used in tbb, which caused the problem
// Use boost instead
//#include <numbers>

#include <boost/math/constants/constants.hpp>
#include <OSL/genclosure.h>

#include "bsdf.h"
#include "base/utils.h"
#include "core/sampling.h"

using OSL::TypeDesc;

enum ClosureID {
    /************
     * Built-in
     ************/
    // BSDF closures
    DiffuseID,
    PhongID,
    OrenNayarID,
    WardID,
    ReflectionID,
    RefractionID,
    TransparentID,
    TranslucentID,

    // Microfacet closures
    MicrofacetID,

    // BSSRDF closures
    SubsurfaceID,

    // Emission closures
    EmissionID,
    BackgroundID,

    // Utility closures
    DebugID,
    HoldoutID,

    /************
     * MaterialX
     ************/

    /************
     * Kazen specific
     ************/
    //GlossyID,

    NumClosureIDs
};

struct EmptyParams      {};
struct DiffuseParams    { OSL::Vec3 N; };

class Diffuse : public BSDF {
public:
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_VECTOR_PARAM(DiffuseParams, N),
            CLOSURE_FINISH_PARAM(DiffuseParams)
        };

        shadingsys.register_closure("diffuse", DiffuseID, params, nullptr, nullptr);
    }

    Diffuse(const DiffuseParams& p)
        : BSDF()
        , params(p)
    {}

    float eval(const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf) const override {
        //pdf = std::max(dot(wi, params.N), 0.f) * std::numbers::inv_pi_v;
        pdf = std::max(dot(wi, static_cast<Vec3f>(params.N)), 0.f) * boost::math::constants::one_div_pi<float>();
        return 1.f;
    }

    float sample(const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wi, float& pdf) const override {
        wi = sample_hemisphere();
        wi = tangent_to_world(wi, sg.N, sg.dPdu, sg.dPdv);
        pdf = std::max(dot(wi, params.N), 0.f) * boost::math::constants::one_div_pi<float>();
        return 1.f;
    }

public:
    DiffuseParams params;
};

struct PhongParams {
    OSL::Vec3 N;
    float exponent;
};

class Phong : public BSDF {

};

class Emission : public BSDF {
public:
    static void register_closure(OSL::ShadingSystem& shadingsys) {
        const OSL::ClosureParam params[] = {
            CLOSURE_FINISH_PARAM(EmptyParams)
        };

        shadingsys.register_closure("emission", EmissionID, params, nullptr, nullptr);
    }

    Emission(const EmptyParams& p)
        : BSDF()
    {}

    float eval(const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf) const override {
        pdf = std::max(dot(wi, sg.N), 0.f) * boost::math::constants::one_div_pi<float>();
        return 1.f;
    }

    float sample(const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wi, float& pdf) const override {
        wi = sample_hemisphere();
        wi = tangent_to_world(wi, sg.N, sg.dPdu, sg.dPdv);
        pdf = std::max(dot(wi, sg.N), 0.f) * boost::math::constants::one_div_pi<float>();
        return 1.f;
    }
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

RGBSpectrum CompositeClosure::sample(const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wi, float& pdf) const {
    float acc = 0;
    RGBSpectrum ret;

    /*
        * The mixture bsdf implementation differs between renderers.
        * In Mitsuba, the sampled component need to multiply an extra bsdf pdf.
        * In testrender of OSL, each compoenent divides an extra tech pdf.
        * Code here removes the extra pdf multiply/divide..
        * TODO : More tests and analytical expected value deduce.
        */

    // An extra 0.9999999 to ensure sampled index don't overflow
    uint idx = sample[0] * 0.9999999f * bsdf_count;
    ret = weights[idx] * (bsdfs[idx]->sample(sg, sample, wi, pdf) / pdfs[idx]);
    pdf *= pdfs[idx];

    // Add up contributions from other bsdfs
    for (int i = 0; i < bsdf_count; i++) {
        if (i == idx) continue;
        float bsdf_pdf = 0;
        RGBSpectrum bsdf_weight = weights[i] * bsdfs[i]->eval(sg, wi, bsdf_pdf);
        power_heuristic(&ret, &pdf, bsdf_weight, bsdf_pdf, pdfs[i]);
    }

    return ret;
};

RGBSpectrum CompositeClosure::eval(const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf) const {
    RGBSpectrum ret;
    pdf = 0;
    for (int i = 0; i < bsdf_count; i++) {
        float bsdf_pdf = 0;
        RGBSpectrum bsdf_weight = weights[i] * bsdfs[i]->eval(sg, wi, bsdf_pdf);
        power_heuristic(&ret, &pdf, bsdf_weight, bsdf_pdf, pdfs[i]);
    }

    return ret;
}

namespace
{
    template <typename ClosureType>
    void register_closure(OSL::ShadingSystem& shadingsys) {
        ClosureType::register_closure(shadingsys);
    }
}

void register_closures(OSL::ShadingSystem *shadingsys) {
    register_closure<Diffuse>(*shadingsys);
    register_closure<Emission>(*shadingsys);
}

void process_closure(ShadingResult& ret, const OSL::ClosureColor *closure, const RGBSpectrum& w, bool light_only) {
    if (!closure)
        return;

    RGBSpectrum cw;
    switch (closure->id) {
        case OSL::ClosureColor::MUL: {
            cw = w * closure->as_mul()->weight;
            process_closure(ret, closure->as_mul()->closure, cw, light_only);
            break;
        }

        case OSL::ClosureColor::ADD: {
            process_closure(ret, closure->as_add()->closureA, w, light_only);
            process_closure(ret, closure->as_add()->closureB, w, light_only);
            break;
        }

        default: {
            const OSL::ClosureComponent *comp = closure->as_comp();
            cw = w * comp->w;

            if (comp->id == EmissionID)
                ret.Le += cw;
            
            if (!light_only) {
                bool status = false;
                switch (comp->id) {
                    case DiffuseID:        status = ret.surface.add_bsdf<Diffuse, DiffuseParams>(cw, *comp->as<DiffuseParams>());
                        break;

                    case EmissionID:       status = ret.surface.add_bsdf<Emission, EmptyParams>(cw, *comp->as<EmptyParams>());
                        break;
                }
                OSL_ASSERT(status && "Invalid closure invoked");
            }

            break;
        }
    }
}
