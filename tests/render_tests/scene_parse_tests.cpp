#include <catch2/catch_all.hpp>

#include "core/scene.h"

using Catch::Approx;

TEST_CASE("Parse test", "[single-file]") {
    Scene scene;
    scene.parse_from_file("../resource/scene/test_scene.xml");

    REQUIRE(scene.film->width == 800);
    REQUIRE(scene.camera->fov == Approx(60.0));
    //REQUIRE(scene.objects[0]->radius == Approx(3));
    //REQUIRE(scene.objects[0]->shader_name == "checker_matte");
}