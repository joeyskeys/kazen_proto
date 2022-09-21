#include <catch2/catch.hpp>

#include <OSL/dual.h>

TEST_CASE("Dual arithmetic", "basic") {
    OSL::Dual<float> x(1, 1);
    auto add_lambda = [](const auto& a, const auto& b) {
        return a + b;
    };

    auto ret = add_lambda(x, 1.f);
    REQUIRE(ret.dx() == Approx(1));

    ret = x * 2;
    REQUIRE(ret.dx() == Approx(2));

    OSL::Dual<float> y(0, 1);
    ret = OSL::sin(y);
    REQUIRE(ret.dx() == Approx(1));

    OSL::Dual<float> a(2, 1), b(2, 1);

    // Perhaps it's a bug?
    // No interface for operations like x^2
    ret = OSL::fast_safe_pow(a, b);
    REQUIRE(ret.dx() == Approx(4));
}
