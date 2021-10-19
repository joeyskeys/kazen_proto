#pragma once

#include <algorithm>

#include "base/vec.h"
#include "base/types.h"
#include "base/spectrum.h"

class BSDF {
public:
    BSDF() {}

    virtual float albedo(const OSL::ShaderGlobals& sg) const {
        return 1;
    }

    virtual float eval(const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf) const = 0;
    virtual float sample(const OSL::ShaderGlobals& sg, Vec3f& wi, float& pdf) const = 0;
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
    bool add_bsdf(const RGBSpectrum& w, const Param& params) {
        if (bsdf_count >= max_closure)
            return false;
        if (byte_count + sizeof(Type) > max_size) return false;

        weights[bsdf_count] = w;
        bsdfs[bsdf_count] = new (pool + byte_count) Type(params);
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

        if ((!cuf_off && weight_sum > 0) || weight_sum) {
            /*
            std::for_each(pdfs.begin(), pdfs.end(), [](float& p) {
                p /= weight_sum
            });
            */
            std::transform(std::excution::par, pdfs.begin(), pdfs.end(), [&weight_sum](auto& pdf){
                pdf /= weight_sum;
            });
        }
    }

    RGBSpectrum eval(const ShaderGlobals& sg, const Vec3f& wi, float& pdf) const {
        RGBSpectrum ret;
        pdf = 0;
        for (int i = 0; i < bsdf_count; i++) {
            float bsdf_pdf = 0;
            RGBSpectrum bsdf_weight = weights[i] * bsdfs[i]->eval(sg, wi, bsdf_pdf);
            // mis update
        }

        return ret;
    }

    RGBSpectrum sample(const ShaderGlobals& sg, Vec3f& wi, float& pdf) const {
        // TODO : pass in a sampler
        float acc = 0;
        RGBSpectrum ret;
        // TODO : sample one bsdf to sample direction
        uint idx = 0; // fake
        ret = bsdfs[idx]->sample(sg, wi, pdf) / pdfs[idx] * weights[idx];

        // Add up contributions from other bsdfs
        for (int i = 0; i < bsdf_count; i++) {
            if (i == idx) continue;
            float other_pdf = 0;
            RGBSpectrum bsdf_weight = weights[i] * bsdfs[i]->eval(sg, wi, bsdf_pdf);
            // mis update
        }

        return ret;
    }


private:
    uint bsdf_count, byte_count;
    std::array<BSDF*, max_closure> bsdfs;
    std::array<float, max_closure> pdfs;
    std::array<RGBSpectrum, max_closure> weights;
    std::array<char, max_size> pool;
}

void register_closures(OSL::ShadingSystem *shadingsys);
void process_closure(const OSL::ClosureColor *Ci, bool light_only);