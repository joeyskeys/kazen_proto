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
    float dtheta = 0.05f;
    float dphi = 0.05f;

    for (int i = 0; i * dtheta < constants::pi<float>(); ++i) {
        auto theta = i * dtheta;
        float c_theta, s_theta;
        sincosf(theta, &s_theta, &c_theta);
        for (int j = 0; j * dphi < constants::two_pi<float>(); ++j) {
            auto phi = j * dphi;
            float c_phi, s_phi;
            sincosf(phi, &s_phi, &c_phi);
            auto L = Vec3f{c_phi * s_theta, c_theta, s_phi * s_theta};
            auto H = base::normalize(V + L);

            auto G1 = mdf.G1(H);
            auto D = mdf.D(H);
            integral += s_theta * D * G1 / std::abs(4.f * V[1]);
        }
    }

    return integral * dtheta * dphi;
}

float fixed_beckmann_white_furnace_test(const float, const float);
float fixed_beckmann_aniso_white_furnace_test(const float, const float, const float);
float fixed_ggx_white_furnace_test(const float, const float);
float fixed_ggx_white_furnace_test(const float, const float, const float);