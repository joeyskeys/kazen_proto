#include <catch2/catch.hpp>

#include "base/vec.h"

using namespace base;

TEST_CASE("Vector operations", "[single-file]") {
    // Arithmetic operators
    Vec3f v1{1, 0, 0}, v2{0, 1, 0}, v3{1, 1, 0}, v4{0, 0, 0};
    REQUIRE(v1 + v2 == v3);
    REQUIRE(v3 - v1 == v2);

    v4 += v1;
    REQUIRE(v4 == v1);
    v4 = v3;
    v4 -= v2;
    REQUIRE(v4 == v1);

    Vec3f v5{2, 2, 0};
    auto v6 = v3 * 2;
    REQUIRE(v5 == v6);
    v6 *= 2;
    Vec3f v7{4, 4, 0};
    REQUIRE(v6 == v7);

    auto v8 = v7 * v1;
    REQUIRE(v8 == Vec3f{4, 0, 0});
    v7 /= Vec3f{2, 2, 1};
    REQUIRE(v7 == v5);

    // Indexing and component getting
    REQUIRE(v8.x() == Approx(4));
    REQUIRE(v8[0] == Approx(4));
    auto data_ptr = v8.data();
    REQUIRE(*data_ptr == Approx(4));

    // Horizontal ops
    v8 = Vec3f{4, 3, 1};
    REQUIRE(max_component(v8) == Approx(4));
    REQUIRE(min_component(v8) == Approx(1));
    REQUIRE(sum(v8) == Approx(8));

    auto v9 = Vec3f{1, 2, 3};
    REQUIRE(length(v9) == Approx(3.7416573867739413));
    REQUIRE(length_squared(v9) == Approx(14));

    // Dot & cross
    v1 = Vec3f{1, 0, 0};
    v2 = Vec3f{0, 1, 0};
    v3 = Vec3f{1, 1, 1};
    v4 = Vec3f{0, 0, 1};
    REQUIRE(dot(v1, v3) == 1);
    REQUIRE(cross(v1, v2) == v4);

    // Normalize
    v1 = Vec3f{1, 1, 1};
    v2 = Vec3f{0.577350269, 0.577350269, 0.577350269};
    REQUIRE(normalize(v1) == v2);

    // Misc
    v1 = Vec3f{1, -1, 0};
    v2 = Vec3f{1, 1, 0};
    REQUIRE(abs(v1) == v2);

    // Comparison
    auto v10 = Vec3f{0, 0, 0};
    REQUIRE(is_zero(v10) == true);
}