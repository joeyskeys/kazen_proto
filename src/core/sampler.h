#pragma once

#include "base/basic_types.h"
#include "base/vec.h"
#include "config.h"

class Sampler {
public:
    void seed(uint x, uint y);
    float randomf();
    Vec2f random2f();
    Vec3f random3f();

private:
#ifdef USE_PCG
    pcg32 m_random;
#endif
};