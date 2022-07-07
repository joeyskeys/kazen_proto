#pragma once

#include "base/basic_types.h"
#include "base/vec.h"
#include "config.h"

class Sampler {
public:
    void    seed(uint x, uint y);
    float   randomf();
    Vec2f   random2f();
    Vec3f   random3f();
    int     randomi(int);

private:
#ifdef USE_PCG
    pcg32 m_random;
#else
    std::uniform_real_distribution<double> m_f_dist(0., 1.);
    std::uniform_int_distribution<> m_i_dist(0);
    std::mt19937 m_gen;
#endif
};