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
    virtual float sample(const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf) const = 0;
};

static const uint max_closure = 8;

class CompositeBSDF {
public:
    CompositeBSDF() : bsdf_count(0) {}

    void compute_pdfs(const OSL::ShaderGlobals& sg, const RGBSpectrum& beta) {
        float w = 1 / beta.sum();
        float weight_sum = 0;
        for (int i = 0; i < bsdf_count; i++) {
            pdfs[i] = dot(weights[i], beta) * bsdfs[i]->albedo(sg) * w;
            weight_sum += pdfs[i];
        }

        std::for_each(pdfs.begin(), pdfs.end(), [](float& p) {
            p /= weight_sum
        });
    }

    RGBSpectrum eval(const ShaderGlobals& sg, const Vec3f& wi, float& pdf) const {

    }

    RGBSpectrum sample(const ShaderGlobals& sg, Vec3f& wi, float& pdf) const {

    }

    template <typename Type, typename Params>
    bool add_bsdf(const RGBSpectrum& w, const Param& params) {
        return true;
    }

private:
    uint bsdf_count;
    std::array<BSDF*, max_closure> bsdfs;
    std::array<float, max_closure> pdfs;
    std::array<RGBSpectrum, max_closure> weights;
}