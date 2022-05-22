// There's a removed feature of cpp20 used in tbb, which caused the problem
// Use boost instead
//#include <numbers>

#include <boost/math/constants/constants.hpp>
#include <OSL/genclosure.h>

#include "base/utils.h"
#include "core/sampling.h"
#include "shading/bsdf.h"
#include "shading/bsdfs.h"

using OSL::TypeDesc;

RGBSpectrum CompositeClosure::sample(const OSL::ShaderGlobals& sg, BSDFSample& sample) const {
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
    auto sp = random3f();
    uint idx = sp[0] * 0.9999999f * bsdf_count;
    auto id = bsdf_ids[idx];
    if (get_sample_func(id) == nullptr)
        return ret;

    ret = weights[idx] * get_sample_func(id)(bsdf_params[idx], sg,
        sample) / pdfs[idx];
    sample.pdf *= pdfs[idx];

    // Add up contributions from other bsdfs
    for (int i = 0; i < bsdf_count; i++) {
        if (i == idx) continue;
        float bsdf_pdf = 0;
        auto other_id = bsdf_ids[i];
        RGBSpectrum bsdf_weight = weights[i] * get_eval_func(other_id)(
            bsdf_params[idx], sg, sample);
        power_heuristic(&ret, &sample.pdf, bsdf_weight, bsdf_pdf, pdfs[i]);
    }

    return ret;
};

RGBSpectrum CompositeClosure::eval(const OSL::ShaderGlobals& sg, BSDFSample& sample) const {
    RGBSpectrum ret;
    float pdf = 0;
    for (int i = 0; i < bsdf_count; i++) {
        auto id = bsdf_ids[i];
        RGBSpectrum bsdf_weight = weights[i] * get_eval_func(id)(
            bsdf_params[i], sg, sample);
        power_heuristic(&ret, &pdf, bsdf_weight, sample.pdf, pdfs[i]);
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
    register_closure<Reflection>(*shadingsys);
    register_closure<Emission>(*shadingsys);
    register_closure<Mirror>(*shadingsys);
    register_closure<Dielectric>(*shadingsys);
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
                    case DiffuseID:         status = ret.surface.add_bsdf<DiffuseParams>(DiffuseID, cw, comp->as<DiffuseParams>());
                        break;

                    case ReflectionID:      status = ret.surface.add_bsdf<ReflectionParams>(ReflectionID, cw, comp->as<ReflectionParams>());
                        break;

                    case EmissionID:        status = ret.surface.add_bsdf<EmptyParams>(EmissionID, cw, comp->as<EmptyParams>());
                        break;

                    case MirrorID:          status = ret.surface.add_bsdf<EmptyParams>(MirrorID, cw, comp->as<EmptyParams>());
                        break;

                    case DielectricID:      status = ret.surface.add_bsdf<DielectricParams>(DielectricID, cw, comp->as<DielectricParams>());
                        break;
                }
                OSL_ASSERT(status && "Invalid closure invoked");
            }

            break;
        }
    }
}
