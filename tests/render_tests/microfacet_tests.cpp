#include <catch2/catch.hpp>

#include "shading/microfacet.h"
#include "shading/white_furnace.h"

TEST_CASE("White furnace test", "[single-file]") {
    float theta_o = 45.f / 180.f * constants::pi<float>();
    auto wi = Vec3f{std::sin(theta_o), std::cos(theta_o), 0};
    auto interface = MicrofacetInterface<BeckmannDist>(wi, 0.5, 0.5);
    auto ret = weak_white_furnace_test(interface, 360, 90);

    REQUIRE(ret == Approx(1.f));
}