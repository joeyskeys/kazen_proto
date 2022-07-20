#pragma once

#include <random>

#include "base/basic_types.h"
#include "base/vec.h"
#include "config.h"

using base::Vec2f;
using base::Vec3f;
using base::Vec4f;

class Sampler {
public:
    Sampler();

    void    seed(uint x, uint y);
    float   randomf();
    Vec2f   random2f();
    Vec3f   random3f();
    Vec4f   random4f();
    int     randomi(int);

private:
#ifdef USE_PCG
    pcg32 m_random;
#else
    std::uniform_real_distribution<double> m_f_dist;
    std::uniform_int_distribution<> m_i_dist;
    std::mt19937 m_gen;
#endif
};