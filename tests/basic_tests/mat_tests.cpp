#include <catch2/catch.hpp>

#include "base/mat.h"

TEST_CASE("Matrix ops", "operators") {
    Mat3f m1 = identity<float, 3>();
    REQUIRE(m1[0] == Vec3f{1, 0, 0});
}