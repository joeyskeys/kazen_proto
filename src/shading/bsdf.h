#pragma once

#include <algorithm>
#include <execution>

#include <OSL/oslexec.h>
#include <OSL/oslclosure.h>

#include "base/vec.h"
#include "core/hitable.h"
#include "core/intersection.h"
#include "core/sampler.h"
#include "core/spectrum.h"
#include "shading/context.h"

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

    // Replicate Nori
    KpMirrorID,
    KpDielectricID,
    KpMicrofacetID,
    KpEmitterID,

    // intrinsic
    KpGlossID,
    KpGlassID,
    KpPrincipleDiffuseID,
    KpPrincipleRetroID,
    KpPrincipleFakeSSID,
    KpPrincipleSheenID,
    KpPrincipleSpecularReflectionID,
    KpPrincipleSpecularRefractionID,
    KpPrincipleClearcoatID,
    KpPrincipleBSSRDFID,

    // BSSRDFs
    KpStandardDipoleID,
    KpBetterDipoleID,

    NumClosureIDs
};

static constexpr uint max_closure = 8;
static constexpr uint max_size = 256 * sizeof(float);

class ScatteringMode {
public:
    enum Mode {
        None        = 0,
        Diffuse     = 1 << 0,
        Specular    = 1 << 1,
        All         = Diffuse | Specular
    };

    static bool has_diffuse(const int modes) {
        return (modes & Diffuse) != 0;
    }

    static bool has_specular(const int modes) {
        return (modes & Specular) != 0;
    }
};

struct BSDFSample {
    // A BSDF sampling parameter pack
    Vec3f wo;
    float pdf;
    float bssrdf_r;
    float bssrdf_idx;
    ScatteringMode::Mode mode;
};

static inline void power_heuristic(RGBSpectrum* w, float* pdf, RGBSpectrum ow, float opdf, float b) {
    // Code copied from OpenShadingLanguage sample
    assert(*pdf >= 0);
    assert(opdf >= 0);
    assert(b >= 0);
    assert(b <= 1);

    if (b > std::numeric_limits<float>::min()) {
        opdf *= b;
        ow *= 1.f / b;
        float mis;
        if (*pdf < opdf)
            mis = 1.f / (1.f + *pdf / opdf);
        else if (opdf < *pdf)
            mis = 1.f - 1.f / (1.f + opdf / *pdf);
        else
            mis = 0.5f;

        *w = *w * (1 - mis) + ow * mis;
        *pdf += opdf;
    }

    assert(*pdf >= 0);
}

class CompositeClosure {
public:
    CompositeClosure()
        : closure_count(0)
        , byte_count(0)
    {}

    template <typename Params>
    bool add_closure(const ClosureID& id, const RGBSpectrum& w, const Params* params) {
        if (closure_count >= max_closure)
            return false;
        if (byte_count + sizeof(Params) > max_size) return false;

        weights[closure_count] = w;
        //bsdfs[closure_count] = new (pool.data() + byte_count) Type(params);
        closure_ids[closure_count] = id;
        closure_params[closure_count] = new (pool.data() + byte_count) Params{};
        //closure_params[closure_count] = params;
        memcpy(closure_params[closure_count], params, sizeof(Params));

        closure_count++;
        byte_count += sizeof(Params);
        return true;
    }

    virtual void compute_pdfs(const OSL::ShaderGlobals& sg, const RGBSpectrum& beta, bool cut_off) {
        float w = 1.f / base::sum(beta);
        float weight_sum = 0;
        for (int i = 0; i < closure_count; i++) {
            //pdfs[i] = dot(weights[i], beta) * bsdfs[i]->albedo(sg) * w;
            pdfs[i] = dot(weights[i], beta) * w;
            weight_sum += pdfs[i];
        }

        if ((!cut_off && weight_sum > 0) || weight_sum > 1) {
            std::for_each(std::execution::par, pdfs.begin(), pdfs.end(), [&weight_sum](float& pdf) {
                pdf /= weight_sum;
            });
            /*
            std::transform(std::execution::par, pdfs.begin(), pdfs.end(), [&weight_sum](auto& pdf) {
                pdf /= weight_sum;
                });
            */
        }
    }

    // Sample type may differ, make these function pure virtual ones
    //virtual RGBSpectrum sample(const OSL::ShaderGlobals& sg, void* sample, const Vec4f& rand) const;
    //virtual RGBSpectrum eval(const OSL::ShaderGlobals& sg, void* sample) const;
    virtual RGBSpectrum sample(ShadingContext*, Sampler*) const;
    virtual RGBSpectrum eval(ShadingContext*) const;

    operator bool() const {
        return closure_count > 0;
    }

protected:
    uint closure_count, byte_count;
    //std::array<BSDF*, max_closure> bsdfs;
    std::array<ClosureID, max_closure> closure_ids;
    std::array<void*, max_closure> closure_params;
    std::array<float, max_closure> pdfs;
    std::array<RGBSpectrum, max_closure> weights;
    std::array<char, max_size> pool;
};

class SurfaceCompositeClosure : public CompositeClosure {

};

class SubsurfaceCompositeClosure : public CompositeClosure {
public:
    void        precompute(ShadingContext*);
    RGBSpectrum sample(ShadingContext*, Sampler*) const override;
    RGBSpectrum eval(ShadingContext*) const override;
};

class EmissionCompositeClosure : public CompositeClosure {

};

class BackgroundCompositeClosure : public CompositeClosure {

};

struct ShadingResult {
    // The implementation in testrender in OSL is reasonable.
    // We want the emission lobe can be sampled later on, so
    // put it into the composite closure. The Le will be set
    // when processing the closure tree and a diffuse EDF will
    // be added into the closure stack for future sampling.
    RGBSpectrum                 Le = RGBSpectrum{0};
    SurfaceCompositeClosure     surface;

    // BSSRDF composite closure is a seperate part, filtered
    // out during process_closure procudure
    SubsurfaceCompositeClosure  bssrdf;
};

struct BSSRDFSample {
    Vec3f po;
    Vec3f wo;
    Frame frame;
    float pdf;
    float axis_prob;
    float pt_prob;
    uint32_t sampled_axis;
    uint32_t sample_cnt;
    Vec3f sigma_a;
    Vec3f sigma_s, sigma_s_prime;
    Vec3f sigma_t, sigma_t_prime;
    Vec3f sigma_tr;
    Vec3f alpha_prime;
    SurfaceCompositeClosure sampled_closure;
    OSL::ShaderGroupRef sampled_shader;
    Vec3f brdf_f;
    float brdf_pdf;
};

void register_closures(OSL::ShadingSystem *shadingsys);
void process_closure(ShadingResult& ret, const OSL::ClosureColor *closure, const RGBSpectrum& w, bool light_only);

// This function follows the signature in OSL testrender
RGBSpectrum process_bg_closure(const OSL::ClosureColor *closure);