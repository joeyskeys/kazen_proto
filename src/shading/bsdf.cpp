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

RGBSpectrum CompositeClosure::sample(const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec4f& sp) const {
    float acc = 0;
    RGBSpectrum ret{0};

    /*
        * The mixture bsdf implementation differs between renderers.
        * In Mitsuba, the sampled component need to multiply an extra bsdf pdf.
        * In testrender of OSL, each compoenent divides an extra tech pdf.
        * Code here removes the extra pdf multiply/divide..
        * TODO : More tests and analytical expected value deduce.
        */

    // An extra 0.9999999 to ensure sampled index don't overflow
    //auto sp = random3f();
    uint idx = sp[3] * 0.9999999f * closure_count;
    auto id = closure_ids[idx];
    if (get_sample_func(id) == nullptr)
        return ret;

    ret = weights[idx] * get_sample_func(id)(closure_params[idx], sg,
        sample, base::head<3>(sp)) / pdfs[idx];
    sample.pdf *= pdfs[idx];

    // Add up contributions from other bsdfs
    for (int i = 0; i < closure_count; i++) {
        if (i == idx) continue;
        float bsdf_pdf = 0;
        auto other_id = closure_ids[i];
        BSDFSample other_sample = sample;
        RGBSpectrum bsdf_weight = weights[i] * get_eval_func(other_id)(
            closure_params[i], sg, other_sample);
        //power_heuristic(&ret, &sample.pdf, bsdf_weight, other_sample.pdf, pdfs[i]);
        ret += bsdf_weight;
        sample.pdf += pdfs[i] * other_sample.pdf;
    }

    return ret;
};

RGBSpectrum CompositeClosure::eval(const OSL::ShaderGlobals& sg, BSDFSample& sample) const {
    RGBSpectrum ret{0};
    float pdf = 0;
    for (int i = 0; i < closure_count; i++) {
        auto id = closure_ids[i];
        RGBSpectrum bsdf_weight = weights[i] * get_eval_func(id)(
            closure_params[i], sg, sample);
        //power_heuristic(&ret, &pdf, bsdf_weight, sample.pdf, pdfs[i]);
        ret += bsdf_weight;
        pdf += pdfs[i] * sample.pdf;
    }

    sample.pdf = pdf;
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
    register_closure<Ward>(*shadingsys);
    register_closure<Reflection>(*shadingsys);
    register_closure<Refraction>(*shadingsys);
    //register_closure<Microfacet>(*shadingsys);
    //register_closure<MicrofacetAniso>(*shadingsys);

    // Microfacet closure is a little bit different
    // FIXME : found a more consistent way to organize the code
    const OSL::ClosureParam params[] = {
        CLOSURE_STRING_PARAM(MicrofacetParams, dist),
        CLOSURE_VECTOR_PARAM(MicrofacetParams, N),
        CLOSURE_VECTOR_PARAM(MicrofacetParams, U),
        CLOSURE_FLOAT_PARAM(MicrofacetParams, xalpha),
        CLOSURE_FLOAT_PARAM(MicrofacetParams, yalpha),
        CLOSURE_FLOAT_PARAM(MicrofacetParams, eta),
        CLOSURE_INT_PARAM(MicrofacetParams, refract),
        CLOSURE_FINISH_PARAM(MicrofacetParams)
    };
    shadingsys->register_closure("microfacet", MicrofacetID, params, nullptr, nullptr);

    register_closure<Emission>(*shadingsys);
    register_closure<Background>(*shadingsys);
    register_closure<KpMirror>(*shadingsys);
    register_closure<KpDielectric>(*shadingsys);
    register_closure<KpMicrofacet>(*shadingsys);
    register_closure<KpEmitter>(*shadingsys);
    register_closure<KpGloss>(*shadingsys);
    register_closure<KpGlass>(*shadingsys);
    register_closure<KpPrincipleDiffuse>(*shadingsys);
    register_closure<KpPrincipleRetro>(*shadingsys);
    register_closure<KpPrincipleFakeSS>(*shadingsys);
    register_closure<KpPrincipleSheen>(*shadingsys);
    register_closure<KpPrincipleSpecularReflection>(*shadingsys);
    register_closure<KpPrincipleClearcoat>(*shadingsys);
}

void process_closure(ShadingResult& ret, const OSL::ClosureColor *closure, const RGBSpectrum& w, bool light_only) {
    static const OSL::ustring u_ggx("ggx");
    static const OSL::ustring u_beckmann("beckmann");
    static const OSL::ustring u_default("default");
    if (!closure)
        return;

    RGBSpectrum cw;
    switch (closure->id) {
        case OSL::ClosureColor::MUL: {
            cw = w * base::to_vec3(closure->as_mul()->weight);
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
            cw = w * base::to_vec3(comp->w);

            if (comp->id == EmissionID || comp->id == KpEmitterID)
                ret.Le += cw;
            
            if (!light_only) {
                bool status = false;
                switch (comp->id) {
                    case DiffuseID:         status = ret.surface.add_closure<DiffuseParams>(DiffuseID, cw, comp->as<DiffuseParams>());
                        break;

                    case WardID:            status = ret.surface.add_closure<WardParams>(WardID, cw, comp->as<WardParams>());
                        break;

                    case ReflectionID:      status = ret.surface.add_closure<ReflectionParams>(ReflectionID, cw, comp->as<ReflectionParams>());
                        break;

                    case RefractionID:      status = ret.surface.add_closure<RefractionParams>(RefractionID, cw, comp->as<RefractionParams>());
                        break;

                    case MicrofacetID: {
                        /*
                        const MicrofacetParams* params = comp->as<MicrofacetParams>();
                        if (params->dist == u_ggx) {
                            switch (params->refract) {
                                case 0: status = ret.surface.add_closure<MicrofacetParams>(MicrofacetGGXReflID, cw, params); break;
                                case 1: status = ret.surface.add_closure<MicrofacetParams>(MicrofacetGGXRefrID, cw, params); break;
                                case 2: status = ret.surface.add_closure<MicrofacetParams>(MicrofacetGGXBothID, cw, params); break;
                                case 0: status = ret.surface.add_closure<Kp
                            }
                        }
                        else if (params->dist == u_beckmann || params->dist == u_default) {
                            switch (params->refract) {
                                case 0: status = ret.surface.add_closure<MicrofacetParams>(MicrofacetBeckmannReflID, cw, params); break;
                                case 1: status = ret.surface.add_closure<MicrofacetParams>(MicrofacetBeckmannRefrID, cw, params); break;
                                case 2: status = ret.surface.add_closure<MicrofacetParams>(MicrofacetBeckmannBothID, cw, params); break;
                            }
                        }
                        break;
                        */
                        const MicrofacetParams* params = comp->as<MicrofacetParams>();
                        switch (params->refract) {
                            case 0: status = ret.surface.add_closure<MicrofacetParams>(KpGlossID, cw, params); break;
                            case 1: status = ret.surface.add_closure<MicrofacetParams>(KpGlassID, cw, params); break;
                            case 2: status = ret.surface.add_closure<MicrofacetParams>(KpGlassID, cw, params); break;
                        }
                        break;
                    }

                    case SubsurfaceID:      status = ret.bssrdf.add_closure<SubsurfaceParams>(SubsurfaceID, cw, comp->as<SubsurfaceParams>());
                        break;

                    case EmissionID:        status = ret.surface.add_closure<EmptyParams>(EmissionID, cw, comp->as<EmptyParams>());
                        break;

                    case BackgroundID:      status = ret.surface.add_closure<EmptyParams>(BackgroundID, cw, comp->as<EmptyParams>());
                        break;

                    case KpMirrorID:        status = ret.surface.add_closure<EmptyParams>(KpMirrorID, cw, comp->as<EmptyParams>());
                        break;

                    case KpDielectricID:    status = ret.surface.add_closure<KpDielectricParams>(KpDielectricID, cw, comp->as<KpDielectricParams>());
                        break;

                    case KpMicrofacetID:    status = ret.surface.add_closure<KpMicrofacetParams>(KpMicrofacetID, cw, comp->as<KpMicrofacetParams>());
                        break;

                    // Weight for emission is different
                    case KpEmitterID:       status = ret.surface.add_closure<KpEmitterParams>(KpEmitterID, RGBSpectrum{1}, comp->as<KpEmitterParams>());
                        break;

                    case KpPrincipleDiffuseID: {
                        status = ret.surface.add_closure<DiffuseParams>(KpPrincipleDiffuseID, cw, comp->as<DiffuseParams>());
                        break;
                    }
                    
                    case KpPrincipleRetroID: {
                        status = ret.surface.add_closure<KpPrincipleRetroParams>(KpPrincipleRetroID, cw, comp->as<KpPrincipleRetroParams>());
                        break;
                    }

                    case KpPrincipleFakeSSID: {
                        status = ret.surface.add_closure<KpPrincipleFakeSSParams>(KpPrincipleFakeSSID, cw, comp->as<KpPrincipleFakeSSParams>());
                        break;
                    }

                    case KpPrincipleSheenID: {
                        status = ret.surface.add_closure<KpPrincipleSheenParams>(KpPrincipleSheenID, cw, comp->as<KpPrincipleSheenParams>());
                        break;
                    }

                    case KpPrincipleSpecularReflectionID: {
                        status = ret.surface.add_closure<KpPrincipleSpecularParams>(KpPrincipleSpecularReflectionID, cw, comp->as<KpPrincipleSpecularParams>());
                        break;
                    }

                    case KpPrincipleClearcoatID: {
                        status = ret.surface.add_closure<KpPrincipleClearcoatParams>(KpPrincipleClearcoatID, cw, comp->as<KpPrincipleClearcoatParams>());
                        break;
                    }
                }
                //OSL_ASSERT(status && "Invalid closure invoked");
                if (!status) {
                    std::cout << "error closure id : " << comp->id << std::endl;
                    throw std::runtime_error(fmt::format("Invalid closure invoked: {}", comp->id));
                }
            }

            break;
        }
    }
}

RGBSpectrum process_bg_closure(const OSL::ClosureColor *closure) {
    if (!closure) return RGBSpectrum{0};
    switch (closure->id) {
        case OSL::ClosureColor::MUL: {
            return base::to_vec3(closure->as_mul()->weight) *
                process_bg_closure(closure->as_mul()->closure);
        }
        case OSL::ClosureColor::ADD: {
            return process_bg_closure(closure->as_add()->closureA) + 
                process_bg_closure(closure->as_add()->closureB);
        }
        case BackgroundID: {
            return closure->as_comp()->w;
        }
    }

    // Should never happen, debug purple color to indicate error
    return RGBSpectrum{1, 0, 1};
}