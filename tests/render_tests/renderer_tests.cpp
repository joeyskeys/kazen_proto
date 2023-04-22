#include <catch2/catch_all.hpp>

#include "core/renderer.h"

using namespace base;

TEST_CASE("Renderer test", "[single-file]") {
    Renderer renderer{5, 16};
    Scene scene;

    scene.set_film(1920, 1080, "./test.png");
    scene.set_camera(Vec3f{0.f, 0.f, 5.f},
        Vec3f{0.f, 4.371e-8f, 4.f},
        Vec3f{0.f, 1.f, 0.f},
        1, 1000, 39.6f);

    std::string lib_path = "/home/joeys/Desktop/softs/kazen/shader";
    scene.set_shader_search_path(lib_path);
    scene.begin_shader_group("test");
    scene.load_oso_shader("surface", "disney_brdf", "layer1", lib_path);
    scene.end_shader_group();

    auto mat = Mat4f::identity();
    auto verts = std::vector<Vec3f>{
        Vec3f{ 1, -1,  1},
        Vec3f{ 1,  1,  1},
        Vec3f{-1, -1,  1},
        Vec3f{-1, -1,  1},
        Vec3f{ 1,  1,  1},
        Vec3f{-1,  1,  1}
    };
    auto norms = std::vector<Vec3f>{
        Vec3f{0, 0, 1},
        Vec3f{0, 0, 1},
        Vec3f{0, 0, 1},
        Vec3f{0, 0, 1},
        Vec3f{0, 0, 1},
        Vec3f{0, 0, 1}
    };
    auto ts = std::vector<Vec2f>{
        Vec2f{0.375f, 0.75f},
        Vec2f{0.625f, 0.75f},
        Vec2f{0.375f, 1.f},
        Vec2f{0.375f, 1.f},
        Vec2f{0.625f, 0.75f},
        Vec2f{0.625f, 1.f}
    };
    auto idx = std::vector<Vec3i>{
        // front
        Vec3i{0, 1, 2},
        Vec3i{3, 4, 5},
    };
    scene.add_mesh(mat, verts, norms, ts, idx, "test");
    scene.build_bvh();

    scene.add_point_light(RGBSpectrum{1.f, 1.f, 1.f},
        Vec3f{0.f, 3.f, 3.f});

    renderer.render(scene, "./test.png");

    REQUIRE(true);
}