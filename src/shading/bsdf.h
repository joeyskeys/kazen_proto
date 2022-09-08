#pragma once

#include <algorithm>
#include <execution>

#include <OSL/oslexec.h>
#include <OSL/oslclosure.h>

#include "base/vec.h"
#include "core/hitable.h"
#include "core/intersection.h"
#include "core/spectrum.h"

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
        : bsdf_count(0)
        , byte_count(0)
    {}

    template <typename Params>
    bool add_bsdf(const ClosureID& id, const RGBSpectrum& w, const Params* params) {
        if (bsdf_count >= max_closure)
            return false;
        if (byte_count + sizeof(Params) > max_size) return false;

        weights[bsdf_count] = w;
        //bsdfs[bsdf_count] = new (pool.data() + byte_count) Type(params);
        bsdf_ids[bsdf_count] = id;
        bsdf_params[bsdf_count] = new (pool.data() + byte_count) Params{};
        //bsdf_params[bsdf_count] = params;
        memcpy(bsdf_params[bsdf_count], params, sizeof(Params));

        bsdf_count++;
        byte_count += sizeof(Params);
        return true;
    }

    virtual void compute_pdfs(const OSL::ShaderGlobals& sg, const RGBSpectrum& beta, bool cut_off) {
        float w = 1.f / base::sum(beta);
        float weight_sum = 0;
        for (int i = 0; i < bsdf_count; i++) {
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

    virtual RGBSpectrum sample(const OSL::ShaderGlobals& sg, BSDFSample& sample, const Vec4f& rand) const;
    virtual RGBSpectrum eval(const OSL::ShaderGlobals& sg, BSDFSample& sample) const;

private:
    uint bsdf_count, byte_count;
    //std::array<BSDF*, max_closure> bsdfs;
    std::array<ClosureID, max_closure> bsdf_ids;
    std::array<void*, max_closure> bsdf_params;
    std::array<float, max_closure> pdfs;
    std::array<RGBSpectrum, max_closure> weights;
    std::array<char, max_size> pool;
};

class SurfaceCompositeClosure : public CompositeClosure {

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
    RGBSpectrum             Le = RGBSpectrum{0};
    SurfaceCompositeClosure surface;
};

void register_closures(OSL::ShadingSystem *shadingsys);
void process_closure(ShadingResult& ret, const OSL::ClosureColor *closure, const RGBSpectrum& w, bool light_only);