#include <catch2/catch_all.hpp>

#include "core/intersection.h"
#include "shading/bssrdfs.h"

using Catch::Approx;

TEST_CASE("BSSRDF test", "[single-file]") {
    KpDipoleParams params;
    params.N = OSL::Vec3(0, 1, 0);
    params.Rd = OSL::Vec3(0.44, 0.22, 0.13);
    params.mfp = OSL::Vec3(0.9, 0.9, 0.9);
    params.max_radius = 2.f;
    params.eta = 1.3f;
    params.g = 0.5f;
    KpDipole::precompute(&params);

    Intersection isect_i;
    isect_i.P = Vec3f(0, 0, 0);
    isect_i.wi = Vec3f(-1, 1, 0).normalized();

    ShadingContext ctx;
    ctx.data = &params;
    ctx.isect_i = &isect_i;
    ctx.isect_o.P = Vec3f(0.2, 0, 0);
    ctx.isect_o.wo = Vec3f(1, 1, 0).normalized();

    auto ret = KpDipole::eval(&ctx);
    std::cout << ret << std::endl;
    REQUIRE((ret[0]< 1 && ret[1] < 1 && ret[2] < 1));
}