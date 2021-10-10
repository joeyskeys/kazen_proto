/*
Currently OSL related part is basically a replication
of testrender in OSL code base.
Try to get a deeper understanding of OSL first.
*/

#pragma once

#include <OSL/oslexec.h>
#include <OSL/oslclosure.h>

#include "base/vec.h"
#include "core/spectrum.h"

enum ClosureID {
    // Referenced from appleseed

    // BSDF closures
    AshikhminShirleyID,
    BlinnID,
    DiffuseID,
    DisneyID,
    OrenNayarID,
    PhongID,
    ReflectionID,
    SheenID,
    HairID,
    TranslucentID,

    // Microfacet closures
    GlossyID,
    MetalID,
    GlassID,
    PlasticID,

    // Emission closures
    EmissionID,

    NumClosureIDs
};

class BSDF {
    virtual float albedo(const OSL::ShaderGlobals& sg) const {
        return 1;
    }

    virtual float sample(const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wi, float& pdf) const = 0;

    virtual float eval  (const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf) const = 0;
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

    void setup(const OSL::ShaderGlobals& sg) {
        
    }

    float sample(const OSL::ShaderGlobals& sg, const Vec3f& sample, Vec3f& wi, float& pdf) override {

    }

    float eval(const OSL::ShaderGlobals& sg, const Vec3f& wi, float& pdf) override {

    }

private:
    constexpr const static int MaxEntries = 8;
    RGBSpectrum albedos[MaxEntries];
    float pdfs[MaxEntries];
    BSDFPtr bsdfs[MaxEntries];
    int bsdf_cnt;
}