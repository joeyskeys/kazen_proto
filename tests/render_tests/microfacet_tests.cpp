#include <catch2/catch.hpp>

#include "shading/microfacet.h"
#include "shading/white_furnace.h"

TEST_CASE("White furnace test", "[single-file]") {
    auto ret = fixed_beckmann_white_furnace_test(0.9f, constants::half_pi<float>() * 0.5f);
    REQUIRE(ret == Approx(1.f).epsilon(1e-2));

    float theta_o = 45.f / 180.f * constants::pi<float>();
    auto wi = Vec3f{std::sin(theta_o), std::cos(theta_o), 0};
    auto interface_beckmann = MicrofacetInterface<BeckmannDist>(wi, 1, 1);
    ret = weak_white_furnace_test(interface_beckmann);
    REQUIRE(ret == Approx(1.f).epsilon(1e-1));

    auto interface_ggx = MicrofacetInterface<GGXDist>(wi, 1, 1);
    ret = weak_white_furnace_test(interface_ggx);
    REQUIRE(ret == Approx(1.f).epsilon(1e-1));
}