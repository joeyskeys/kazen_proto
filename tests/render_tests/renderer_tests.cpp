#include <catch2/catch_all.hpp>

#include "core/renderer.h"

using namespace base;

TEST_CASE("Renderer test", "[single-file]") {
    Renderer renderer{5, 16};
    Scene scene;

    scene.set_film(800, 600, "./test.png");
    scene.set_camera(Vec3f{7.3588915f, -6.92571f, 4.9583092f},
        Vec3f{6.7073331f, -6.3116202f, 4.51303768f},
        Vec3f{-0.324013472f, 0.30542084575f, 0.89539564f},
        1, 1000, 39.6f);

    std::string lib_path = "/home/joeys/Desktop/softs/kazen/shader";
    scene.set_shader_search_path(lib_path);
    scene.begin_shader_group("test");
    scene.load_oso_shader("surface", "disney_brdf", "layer1", lib_path);
    scene.end_shader_group();

    auto mat = Mat4f::identity();
    auto verts = std::vector<Vec3f>{
        Vec3f{-1, -1,  1},
        Vec3f{ 1, -1,  1},
        Vec3f{ 1,  1,  1},
        Vec3f{-1,  1,  1},
        Vec3f{-1, -1, -1},
        Vec3f{ 1, -1, -1},
        Vec3f{ 1,  1, -1},
        Vec3f{-1,  1, -1}
    };
    auto norms = std::vector<Vec3f>{
        Vec3f{0, 1, 0},
        Vec3f{0, 1, 0},
        Vec3f{0, 1, 0},
        Vec3f{0, 1, 0},
        Vec3f{0, 1, 0},
        Vec3f{0, 1, 0},
        Vec3f{0, 1, 0},
        Vec3f{0, 1, 0}
    };
    auto ts = std::vector<Vec2f>{
        Vec2f{0, 1},
        Vec2f{1, 0},
        Vec2f{0, 1},
        Vec2f{1, 0},
        Vec2f{0, 1},
        Vec2f{1, 0},
        Vec2f{0, 1},
        Vec2f{1, 0}
    };
    auto idx = std::vector<Vec3i>{
        // front
        Vec3i{0, 1, 2},
        Vec3i{0, 2, 3},
        // back
        Vec3i{4, 7, 6},
        Vec3i{4, 6, 5},
        // up
        Vec3i{3, 2, 6},
        Vec3i{3, 6, 7},
        // down
        Vec3i{0, 4, 1},
        Vec3i{1, 4, 5},
        // left
        Vec3i{0, 3, 7},
        Vec3i{0, 7, 4},
        // right
        Vec3i{1, 5, 6},
        Vec3i{1, 6, 2}
    };
    scene.add_mesh(mat, verts, norms, ts, idx, "test");

    scene.add_point_light(RGBSpectrum{1.f, 1.f, 1.f},
        Vec3f{4.07624531f, 1.005453944f, 5.903862f});

    renderer.render(scene, "./test.png");

    REQUIRE(true);
}