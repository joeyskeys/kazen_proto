#include <vector>

#include <catch2/catch_all.hpp>

#include "base/mat.h"
#include "base/vec.h"
#include "core/scene.h"

using Catch::Approx;
using namespace base;

TEST_CASE("Scene API test", "[single-file]") {
    Scene scene;

    Mat4f m{};
    Vec3f v1{1, 2, 3};
    Vec2f v2{1, 2};
    Vec3i v3{1, 2, 3};

    auto verts = std::vector<Vec3f>();
    verts.push_back(v1);
    auto normals = std::vector<Vec3f>();
    normals.push_back(v1);
    auto uvs = std::vector<Vec2f>();
    uvs.push_back(v2);
    auto faces = std::vector<Vec3i>();
    faces.push_back(v3);
    scene.add_mesh(m, verts, normals, uvs, faces, "test", "test", false);
}