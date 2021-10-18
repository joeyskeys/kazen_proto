/*
Currently OSL related part is basically a replication
of testrender in OSL code base.
Try to get a deeper understanding of OSL first.
*/

#pragma once

#include <algorithm>
#include <array>
#include <excution>

#include <OSL/oslexec.h>
#include <OSL/oslclosure.h>

#include "base/vec.h"
#include "core/spectrum.h"

enum ClosureID {
    // Referenced from appleseed

    // BSDF closures
    //AshikhminShirleyID,
    //BlinnID,
    DiffuseID,
    //DisneyID,
    OrenNayarID,
    //PhongID,
    //ReflectionID,
    //SheenID,
    //HairID,
    //TranslucentID,

    // Microfacet closures
    //GlossyID,
    //MetalID,
    //GlassID,
    //PlasticID,

    // Emission closures
    EmissionID,

    NumClosureIDs
};

class BSDF {
    virtual float albedo(const OSL::ShaderGlobals& sg) const {
        return 1;
    }

    virtual RGBSpectrum sample(const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wi, float& pdf) const = 0;

    virtual RGBSpectrum eval  (const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf) const = 0;
};

using BSDFPtr = BSDF*;

class CompositeBSDF {
public:
    CompositeBSDF()
        : bsdf_cnt(0)
    {}

    template <typename Type, typename Params>
    bool add_bsdf(const RGBSpectrum& a, const Params& params) {
        if (bsdf_cnt >= MaxEntries)
            return false;

        albedos[bsdf_cnt] = a;
        bsdfs[bsdf_cnt] = new Type(params);
        bsdf_cnt++;
        return true;
    }

    void setup(const OSL::ShaderGlobals& sg, const RGBSpectrum& beta, bool cut_off) {
        // No Russian-Roulette cut off now
        float w = 1.f / (beta.x + beta.y + beta.z);
        float pdf_sum = 0.f;
        for (int i = 0; i < bsdf_cnt; i++) {
            pdfs[i] = dot(albedos[i], beta) * bsdfs[i]->albedo(sg) * w;
            pdf_sum += pdfs[i];
        }
        if ((!cut_off && pdf_sum > 0) || pdf_sum > 1.f)
            //for (int i = 0; i < bsdf_cnt; i++)
                //pdfs[i] /= pdf_sum;
            std::transform(std::execution::par, pdfs.begin(), pdfs.end(), [&pdf_sum](auto& pdf){
                pdf /= pdf_sum;
            });
    }

    RGBSpectrum sample(const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wi, float& pdf) override {
        float accum = 0;
        for (int i = 0; i < bsdf_cnt; i++) {
            if (sample[0] < (pdfs[i] + accum)) {
                RGBSpectrum ret = albedos[i] * (bsdfs[i]->sample(sg, sample, wi, pdf) / pdfs[i]);
                pdf *= pdfs[i];

                for (int j = 0; j < bsdf_cnt; j++) {
                    if (i == j) continue;
                    float bsdf_pdf = 0;
                    RGBSpectrum bsdf_albedo = albedos[j] * bsdfs[j]->eval(sg, wi, bsdf_pdf);
                    // MIS
                }

                return ret;
            }
            accum += pdfs[i];
        }
        return RGBSpectrum{};
    }

    RGBSpectrum eval(const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf) override {
        RGBSpectrum ret{};
        pdf = 0;

        for(int i = 0; i < bsdf_cnt; i++) {
            float bsdf_pdf = 0;
            RGBSpectrum bsdf_weight = albedos[i] * bsdfs[i]->eval(sg, wi, bsdf_pdf);
            // MIS
        }

        return result;
    }

private:
    constexpr const static int MaxEntries = 8;
    std::array<RGBSpectrum, MaxEntries> albedos;
    std::array<float, MaxEntries> pdfs;
    std::array<BSDFPtr, MaxEntries> bsdfs;
    int bsdf_cnt;
}