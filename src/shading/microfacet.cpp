#include "shading/microfacet.h"

Vec3f BeckmannMDF(const Vec2f& sample, float alpha) {
    auto phi = constants::two_pi<float>() * sample.x();
    auto theta = std::atan(std::sqrt(-alpha * alpha * std::log(sample.y())));
    auto sin_theta = std::sin(theta);
    return Vec3f(sin_theta * std::cos(phi), sin_theta * std::sin(phi), std::cos(theta));
}

float BeckmannPDF(const Vec3f& m, float alpha) {
    auto cos_theta_i = cos_theta(m);
    if (cos_theta_i <= 0.)
        return 0.f;

    auto azimuthal = constants::one_div_pi<float>();
    auto alpha2 = alpha * alpha;
    auto tan_theta_i = tan_theta(m);
    auto longitudinal = std::exp((-tan_theta_i * tan_theta_i) / alpha2)
                        / //-------------------------------------------
                                (alpha2 * std::pow(cos_theta_i, 3));

    return azimuthal * longitudinal;
}

float G1(const Vec3f& wh, const Vec3f& wv, float alpha) {
    if (dot(wv, wh) / cos_theta(wv) <= 0.f)
        return 0.f;

    auto b = 1. / (alpha * tan_theta(wv));
    if (b >= 1.6f)
        return 1.f;
    else {
        auto b2 = b * b;
        return (3.535 * b + 2.181 * b2)
            / //--------------------------
            (1. + 2.276 * b + 2.577 * b2);
    }
}