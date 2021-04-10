#pragma once

#include "spectrum.h"
#include "material.h"

class Light {
public:
    RGBSpectrum sample_l(const Intersection& isect, const Vec2f& u, Vec3f& wi, float& p) const = 0;
    float pdf(const Intersection& isect, const Vec3f& wi) const = 0;

    // members
    bool is_delta = true;
};