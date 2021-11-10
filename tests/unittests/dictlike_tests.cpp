
#include <catch2/catch_test_macro.hpp>

#include "core/camera.h"

/*
// This's not going to work
// Use runtime version..
TEST_CASE("Member location equality", "[single-file]") {
    Camera cam {
        Vec3f{7.f, 8.f, 15.f},
        Vec3f{0.f, 3.f, -20.f},
        Vec3f{0.f, 1.f, 0.f},
        1.f,
        1000.f,
        60.f,
        &film
    };

    REQUIRE(cam.address_of("position") == cam.runtime_address_of("position"));
    REQUIRE(cam.address_of("lookat") == cam.runtime_address_of("lookat"));
    REQUIRE(cam.address_of("near") == cam.runtime_address_of("near"));
}
*/