#pragma once

#include <boost/math/constants/constants.hpp>

#include "base/vec.h"
#include "shading/bsdfs.h"

using base::Vec3f;

namespace constants = boost::math::constants;

template <typename MDFInterface>
float weak_white_furnace_test(const MDFInterface& mdf,
    const uint32_t phi_span, const uint32_t theta_span)
{
    auto V = mdf.wi;
    float integral = 0.f;

    for (int i = 0; i < theta_span; ++i) {
        auto theta = i / theta_span * constants::half_pi<float>();
        double c_theta, s_theta;
        sincos(theta, &c_theta, &s_theta);
        for (int j = 0; j < phi_span; ++j) {
            auto phi = j / phi_span * constants::two_pi<float>();
            double c_phi, s_phi;
            sincos(phi, &c_phi, &s_phi);
            auto L = Vec3f{c_phi * s_theta, c_theta, s_phi * s_theta};
            auto H = base::normalize(V + L);

            auto G1 = mdf.G1(H);
            auto D = mdf.D(H);
            integral += s_theta * D * G1 / std::abs(4.f * V[1]);
        }
    }

    return integral / theta_span / phi_span;
}