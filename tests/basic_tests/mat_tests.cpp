#include <catch2/catch.hpp>

#include "base/mat.h"

TEST_CASE("Matrix construction", "construction") {
    Mat3f m1 = identity<float, 3>();
    REQUIRE(m1[0] == Vec3f{1, 0, 0});
}

TEST_CASE("Matrix ops", "operation") {
    Mat3f m1 = Mat3f{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    auto m2 = transpose(m1);
    // The index operator returns column
    REQUIRE(m1[0] == Vec3f{1, 4, 7});
    REQUIRE(m2[0] == Vec3f{1, 2, 3});
    m1 = Mat3f {
        1, 2, -1,
        2, 1, 2,
        -1, 2, 1
    };
    m2 = inverse(m1);
    REQUIRE(m2[0] == Vec3f{0.1875, 0.25, -0.3125});
}

TEST_CASE("Matrix multiplication", "multiplication") {
    Mat3f m1 = {
        2, 3, 1,
        7, 4, 1,
        9, -2, 1
    };
    Mat3f m2 = {
        9, -2, -1,
        5, 7, 3,
        8, 1, 0
    };
    auto m3 = m1 * m2;
    REQUIRE(m3[0] == Vec3f{41, 91, 79});

    Vec3f v1 = {1, 2, 3};
    auto m4 = m1 * v1;
    REQUIRE(m4 == Vec3f{11, 18, 8});
}

TEST_CASE("Matrix transform", "transform") {
    auto m1 = translate3f(Vec3f{1, 2, 3});
    REQUIRE(m1[3] == Vec4f{1, 2, 3, 1});

    auto m2 = rotate3f(Vec3f{0, 0, 1}, 90);
    REQUIRE(m2[0][1] == Approx(1));
    REQUIRE(m2[1][0] == Approx(-1));

    auto m3 = scale3f(Vec3f{0.5, 0.5, 0.5});
    REQUIRE(m3[0] == Vec4f{0.5, 0, 0, 0});
}
