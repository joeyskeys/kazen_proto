#include <catch2/catch_test_macro.hpp>

#include "core/scene.h"

TEST_CASE("Parse test", "[single-file]") {
    Scene scene;
    scene.parse_from_file("../resource/scene/test_scene.xml");
}