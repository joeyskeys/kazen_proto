#pragma once

#include <algorithm>
#include <execution>

#include <OSL/oslexec.h>
#include <OSL/oslclosure.h>

#include "base/vec.h"
#include "base/types.h"
#include "core/spectrum.h"

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

class CompositeBSDF {
public:
    CompositeBSDF()
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

    void compute_pdfs(const OSL::ShaderGlobals& sg, const RGBSpectrum& beta, bool cut_off) {
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

    RGBSpectrum eval(const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf) const {
        RGBSpectrum ret;
        pdf = 0;
        for (int i = 0; i < bsdf_count; i++) {
            float bsdf_pdf = 0;
            //ret += weights[i] * bsdfs[i]->eval(sg, wi, bsdf_pdf);
            //pdf += bsdf_pdf * pdfs[i];
            RGBSpectrum bsdf_weight = weights[i] * bsdfs[i]->eval(sg, wi, bsdf_pdf);
            power_heuristic(&ret, &pdf, bsdf_weight, bsdf_pdf, pdfs[i]);
        }

        return ret;
    }

    RGBSpectrum sample(const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wi, float& pdf) const {
        float acc = 0;
        RGBSpectrum ret;

        /*
         * The mixture bsdf implementation differs between renderers.
         * In Mitsuba, the sampled component need to multiply an extra bsdf pdf.
         * In testrender of OSL, each compoenent divides an extra tech pdf.
         * Code here removes the extra pdf multiply/divide..
         * TODO : More tests and analytical expected value deduce.
         */

        uint idx = sample[0] * bsdf_count;
        ret = weights[idx] * (bsdfs[idx]->sample(sg, sample, wi, pdf) / pdfs[idx]);
        pdf *= pdfs[idx];

        // Add up contributions from other bsdfs
        for (int i = 0; i < bsdf_count; i++) {
            if (i == idx) continue;
            //float other_pdf = 0;
            //ret += weights[i] * bsdfs[i]->eval(sg, wi, other_pdf);
            //pdf += other_pdf * pdfs[i];
            float bsdf_pdf = 0;
            RGBSpectrum bsdf_weight = weights[i] * bsdfs[i]->eval(sg, wi, bsdf_pdf);
            power_heuristic(&ret, &pdf, bsdf_weight, bsdf_pdf, pdfs[i]);
        }

        return ret;
    }

private:
    uint bsdf_count, byte_count;
    std::array<BSDF*, max_closure> bsdfs;
    std::array<float, max_closure> pdfs;
    std::array<RGBSpectrum, max_closure> weights;
    std::array<char, max_size> pool;
};

struct ShadingResult {
    RGBSpectrum Le;
    CompositeBSDF bsdf;
};

void register_closures(OSL::ShadingSystem *shadingsys);
void process_closure(ShadingResult& ret, const OSL::ClosureColor *closure, const RGBSpectrum& w, bool light_only);