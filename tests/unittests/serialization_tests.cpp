#include <string>

#include <catch2/catch.hpp>

#include "core/camera.h"

using namespace std::string_literals;

TEST_CASE("Member reflect", "[single-file") {
    Camera cam{};
    Person p{"Joey", 30};

    auto fov = hana::at_key(cam, BOOST_HANA_STRING("fov"));

    REQUIRE(fov == 60);
}