#pragma once

#include <algorithm>
#include <execution>

#include <OSL/oslexec.h>
#include <OSL/oslclosure.h>

#include "base/vec.h"
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
    //GlossyID,

    NumClosureIDs
};

struct EmptyParams      {};
struct DiffuseParams    { OSL::Vec3 N; };
struct PhongParams {
    OSL::Vec3 N;
    float exponent;
};

class BSDF {
public:
    BSDF() {}

    virtual float albedo(const OSL::ShaderGlobals& sg) const {
        return 1;
    }

    virtual float eval(const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf) const = 0;
    virtual float sample(const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wi, float& pdf) const = 0;
};

static constexpr uint max_closure = 8;
static constexpr uint max_size = 256 * sizeof(float);

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

    template <typename Type, typename Params>
    bool add_bsdf(const RGBSpectrum& w, const Params& params) {
        if (bsdf_count >= max_closure)
            return false;
        if (byte_count + sizeof(Type) > max_size) return false;

        weights[bsdf_count] = w;
        bsdfs[bsdf_count] = new (pool.data() + byte_count) Type(params);
        bsdf_count++;
        byte_count += sizeof(Type);
        return true;
    }

    virtual void compute_pdfs(const OSL::ShaderGlobals& sg, const RGBSpectrum& beta, bool cut_off) {
        float w = 1.f / beta.sum();
        float weight_sum = 0;
        for (int i = 0; i < bsdf_count; i++) {
            pdfs[i] = dot(weights[i], beta) * bsdfs[i]->albedo(sg) * w;
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

    virtual RGBSpectrum sample(const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wi, float& pdf) const;
    virtual RGBSpectrum eval(const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf) const;

private:
    uint bsdf_count, byte_count;
    std::array<BSDF*, max_closure> bsdfs;
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
    // The implementation in testrender in OSL in reasonable.
    // We want the emission lobe can be sampled later on, so
    // put it into the composite closure. The Le will be set
    // when processing the closure tree and a diffuse EDF will
    // be added into the closure stack for future sampling.
    RGBSpectrum             Le;
    SurfaceCompositeClosure surface;
};

void register_closures(OSL::ShadingSystem *shadingsys);
void process_closure(ShadingResult& ret, const OSL::ClosureColor *closure, const RGBSpectrum& w, bool light_only);