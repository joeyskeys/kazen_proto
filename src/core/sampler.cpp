#include <cmath>

#include "sampler.h"

Sampler::Sampler()
#ifdef USE_PCG
{}
#else
    : m_f_dist(0., 1.)
    , m_i_dist(0)
{}
#endif

void Sampler::seed(uint x, uint y) {
#ifdef USE_PCG
    m_random.seed(x, y);
#else
    m_gen.seed(std::pow(10, std::floor(std::log10(x) + 1) + y));
#endif
}

float Sampler::randomf() {
#ifdef USE_PCG
    return m_random.nextFloat();
#else
    return m_f_dist(m_gen);
#endif
}

Vec2f Sampler::random2f() {
#ifdef USE_PCG
    return Vec2f{m_random.nextFloat(), m_random.nextFloat()};
#else
    return Vec2f{m_f_dist(m_gen), m_f_dist(m_gen)};
#endif
}

Vec3f Sampler::random3f() {
#ifdef USE_PCG
    return Vec3f{m_random.nextFloat(), m_random.nextFloat(), m_random.nextFloat()};
#else
    return Vec3f{m_f_dist(m_gen), m_f_dist(m_gen), m_f_dist(m_gen)};
#endif
}

int Sampler::randomi(int range) {
#ifdef USE_PCG
    return m_random.nextUInt(range);
#else
    m_i_dist = std::uniform_int_distribution<>(0, range);
    return m_i_dist(m_gen);
#endif
}