#include "shading/white_furnace.h"

float fixed_beckmann_white_furnace_test(const float alpha, const float theta_o) {
    float s_theta_o, c_theta_o, s_theta_i, c_theta_i, s_phi, c_phi;
    sincosf(theta_o, &s_theta_o, &c_theta_o);
    auto V = Vec3f{s_theta_o, 0, c_theta_o};
    auto a  = 1.f / (alpha * std::tan(theta_o));
    auto a2 = a * a;
    auto lambda = 0.f;
    if (a < 1.6f)
        lambda = (1 - 1.259f * a + 0.396f * a2) / (3.535f * a + 2.181f * a2);
    auto G = 1.f / (1.f + lambda);

    float integral = 0.f;
    float dtheta = 0.05;
    float dphi = 0.05;
    for (int i = 0; i * dtheta < constants::pi<float>(); ++i) {
        auto theta_i = i * dtheta;
        for (int j = 0; j * dphi < constants::two_pi<float>(); ++j) {
            auto phi = j * dphi;
            sincosf(theta_i, &s_theta_i, &c_theta_i);
            sincosf(phi, &s_phi, &c_phi);
            auto L = Vec3f{c_phi * s_theta_i, s_phi * s_theta_i, c_theta_i};
            auto H = base::normalize(V + L);
            if (H[2] <= 0)
                continue;

            float theta_h = std::acos(H[2]);
            float D = std::exp(-std::pow(std::tan(theta_h) / alpha, 2)) /
                (constants::pi<float>() * alpha * alpha * std::pow(H[2], 4));
            integral += s_theta_i * D * G / std::abs(4 * V[2]);
        }
    }

    return integral * dtheta * dphi;
}

float fixed_beckmann_aniso_white_furnace_test(const float alpha_x, const float alpha_y,
    const float theta_o)
{
    return 1.f;
}

float fixed_ggx_white_furnace_test(const float alpha, const float theta_o) {
    return 1.f;
}

float fixed_ggx_white_furnace_test(const float alpha_x, const float alpha_y,
    const float theta_o)
{
    return 1.f;
}