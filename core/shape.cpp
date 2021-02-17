#include <filesystem>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "shape.h"
#include "ray.h"
#include "transform.h"

namespace fs = std::filesystem; 

bool Sphere::intersect(Ray& r, Intersection& isect) const {
    auto r_local = world_to_local.apply(r);

    auto oc = r_local.origin - center;
    auto a = r_local.direction.length_squared();
    auto half_b = dot(oc, r_local.direction);
    auto c = oc.length_squared() - radius * radius;
    auto discriminant = half_b * half_b - a * c;

    if (discriminant < 0.f)
        return false;

    auto t0 = (-half_b - sqrtf(discriminant)) / a;
    auto t1 = (-half_b + sqrtf(discriminant)) / a;

    if (t0 > r.tmax || t1 < r.tmin) return false;
    float t = t0;
    isect.backface = false;
    if (t <= r.tmin) {
        t = t1;
        isect.backface = true;
        if (t > r.tmax) return false;
    }

    // Calculate in local space
    isect.ray_t = t;
    // small bias to avoid self intersection
    isect.position = r_local.at(t) * 1.00001f;
    isect.normal = (r_local.at(t) - center) / radius;
    if (isect.backface)
        isect.normal = -isect.normal;

    if (isect.normal == Vec3f(0.f, 1.f, 0.f))
        isect.tangent = normalize(Vec3f{0.f, 0.f, -1.f}.cross(isect.normal));
    else
        isect.tangent = normalize(Vec3f{0.f, 1.f, 0.f}.cross(isect.normal));

    isect.bitangent = normalize(isect.tangent.cross(isect.normal));
    isect.mat = mat;
    isect.obj_id = obj_id;

    // Transform back to world space
    isect = local_to_world.apply(isect);
    
    return true;
}

static bool moller_trumbore_intersect(const Ray& r, const Vec3f* verts, Intersection& isect) {
    Vec3f v1v0 = verts[1] - verts[0];
    Vec3f v2v0 = verts[2] - verts[0];
    Vec3f pvec = cross(r.direction, v2v0);
    float det = dot(v1v0, pvec);

    if (det < 0.000001) return false;

    float det_inv = 1.f / det;
    Vec3f tvec = r.origin - verts[0];
    float u = dot(tvec, pvec) * det_inv;
    if (u < 0.f || u > 1.f) return false;

    Vec3f qvec = cross(tvec, v1v0);
    float v = dot(r.direction, qvec) * det_inv; 
    if (v < 0.f || u + v > 1.f) return false;

    float t = dot(v2v0, qvec) * det_inv;

    isect.position = r.at(t);
    isect.normal = cross(v1v0, v2v0).normalized();
    isect.ray_t = t;
    isect.backface = t < 0.f ? true : false;

    return true;
}

bool Triangle::intersect(Ray& r, Intersection& isect) const {
    auto local_r = world_to_local.apply(r);

    auto ret = moller_trumbore_intersect(local_r, verts, isect);
    if (ret) {
        isect.mat = mat;
        isect = local_to_world.apply(isect);
    }

    return ret;
}

bool TriangleMesh::intersect(Ray& r, Intersection& isect) const {
    auto local_r = world_to_local.apply(r);
    bool hit = false;

    for (auto& idx : indice) {
        // TODO : compare the performance difference between
        // constructing triangle on the fly and converting it
        // to triangle list.
        Vec3f tri[3];
        tri[0] = verts[idx[0]];
        tri[1] = verts[idx[1]];
        tri[2] = verts[idx[2]];
        hit = moller_trumbore_intersect(local_r, tri, isect);
        if (hit) {
            isect.mat = mat;
            isect = local_to_world.apply(isect);
            return true;
        }
    }

    return false;
}

static TriangleMesh process_mesh(aiMesh* mesh, const aiScene* scene) {
    std::vector<Vec3f> vs(mesh->mNumVertices);
    std::vector<Vec3i> idx(mesh->mNumFaces);

    for (uint i = 0; i < mesh->mNumVertices; i++) {
        vs[i][0] = mesh->mVertices[i].x;
        vs[i][1] = mesh->mVertices[i].y;
        vs[i][2] = mesh->mVertices[i].z;
    }

    for (uint i = 0; i < mesh->mNumFaces; i++) {
        idx[i][0] = mesh->mFaces[i].mIndices[0];
        idx[i][1] = mesh->mFaces[i].mIndices[1];
        idx[i][2] = mesh->mFaces[i].mIndices[2];
    }

    return TriangleMesh(Transform{}, std::move(vs), std::move(idx));
}

static void process_node(aiNode* node, const aiScene* scene, std::vector<TriangleMesh>& meshes) {
    for (uint i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        meshes.emplace_back(process_mesh(mesh, scene));
    }

    for (uint i = 0; i < node->mNumChildren; i++) {
        process_node(node->mChildren[i], scene, meshes);
    }
}

std::vector<TriangleMesh> load_triangle_mesh(const std::string& file_path) {
    std::vector<TriangleMesh> meshes;

    if (!fs::exists(file_path))
        return meshes;

    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(file_path, aiProcess_Triangulate);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        return meshes;
    }

    process_node(scene->mRootNode, scene, meshes);

    return meshes;
}