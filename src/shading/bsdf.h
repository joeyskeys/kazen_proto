#pragma once

#include <algorithm>
#include <execution>

#include <OSL/oslexec.h>

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
        float w = 1 / beta.sum();
        float weight_sum = 0;
        for (int i = 0; i < bsdf_count; i++) {
            pdfs[i] = dot(weights[i], beta) * bsdfs[i]->albedo(sg) * w;
            weight_sum += pdfs[i];
        }

        if ((!cut_off && weight_sum > 0) || weight_sum) {
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
            ret += weights[i] * bsdfs[i]->eval(sg, wi, bsdf_pdf);
            pdf += bsdf_pdf * pdfs[i];
        }

        return ret / pdf;
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
        ret = bsdfs[idx]->sample(sg, sample, wi, pdf) * weights[idx];
        pdf *= pdfs[idx];

        // Add up contributions from other bsdfs
        for (int i = 0; i < bsdf_count; i++) {
            if (i == idx) continue;
            float other_pdf = 0;
            ret += weights[i] * bsdfs[i]->eval(sg, wi, other_pdf);
            pdf += other_pdf * pdfs[i];
        }

        return ret / pdf;
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