#include <boost/filesystem.hpp>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "shape.h"
#include "ray.h"
#include "transform.h"

namespace fs = boost::filesystem;

void Shape::print_bound() const {
    std::cout << "id : " << obj_id << std::endl;
    Hitable::print_bound();
}

bool Sphere::intersect(const Ray& r, Intersection& isect) const {
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
    isect.shader_name = shader_name;
    
    return true;
}

bool Sphere::intersect(const Ray& r, float& t) const {
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
    float tmp_t = t0;
    if (tmp_t <= r.tmin) {
        tmp_t = t1;
        if (tmp_t > r.tmax) return false;
    }

    // Calculate in local space
    t = tmp_t;
    
    return true;
}

AABBf Sphere::bbox() const {
    auto center_in_world = local_to_world.apply(center);
    auto radius_vec = Vec3f{radius, radius, radius};
    return AABBf{center_in_world - radius_vec, center_in_world + radius_vec};
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

    isect.position = r.at(t - 0.00001f);
    isect.normal = cross(v1v0, v2v0).normalized();
    // TODO : calculate dpdu & dpdv.
    isect.tangent = v1v0.normalized();
    isect.bitangent = cross(isect.tangent, isect.normal);
    isect.ray_t = t;
    isect.backface = t < 0.f ? true : false;

    return true;
}

void* Sphere::address_of(const std::string& name) {
    if (name == "radius")
        return &radius;
    else if (name == "center")
        return &center;
    else if (name == "shader_name")
        return &shader_name;
    return nullptr;
}

bool Triangle::intersect(const Ray& r, Intersection& isect) const {
    auto local_r = world_to_local.apply(r);

    auto ret = moller_trumbore_intersect(local_r, verts, isect);
    if (ret && !isect.backface) {
        isect.mat = mat;
        isect.obj_id = obj_id;
        isect = local_to_world.apply(isect);
        isect.shader_name = shader_name;
        return true;
    }

    return false;
}

static bool moller_trumbore_intersect(const Ray& r, const Vec3f* verts, float& t) {
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

    t = dot(v2v0, qvec) * det_inv;

    return true;
}

bool Triangle::intersect(const Ray& r, float& t) const {
    auto local_r = world_to_local.apply(r);

    if (moller_trumbore_intersect(local_r, verts, t)) {
        // Not considering backfacing here
        return true;
    }

    return false;
}

static inline AABBf bbox_of_triangle(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2) {
    return bound_union(AABBf{vec_min(v0, v1), vec_max(v0, v1)}, v2);
}

AABBf Triangle::bbox() const {
    auto v0_in_world = local_to_world.apply(verts[0]);
    auto v1_in_world = local_to_world.apply(verts[1]);
    auto v2_in_world = local_to_world.apply(verts[2]);
    return bbox_of_triangle(v0_in_world, v1_in_world, v2_in_world);
}

void* Triangle::address_of(const std::string& name) {
    if (name == "verta")
        return &verts[0];
    else if (name == "vertb")
        return &verts[1];
    else if (name == "vertc")
        return &verts[2];
    else if (name == "shader_name")
        return &shader_name;
    return nullptr;
}

bool TriangleMesh::intersect(const Ray& r, Intersection& isect) const {
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
        if (hit && !isect.backface) {
            isect.mat = mat;
            isect.obj_id = 2;
            isect.shader_name = shader_name;
            isect = local_to_world.apply(isect);
            return true;
        }
    }

    return false;
}

bool TriangleMesh::intersect(const Ray& r, float& t) const {
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
        if (moller_trumbore_intersect(local_r, tri, t)) {
            return true;
        }
    }

    return false;
}

AABBf TriangleMesh::bbox() const {
    AABBf box;
    for (auto& idx : indice) {
        auto v0 = local_to_world.apply(verts[idx[0]]);
        auto v1 = local_to_world.apply(verts[idx[1]]);
        auto v2 = local_to_world.apply(verts[idx[2]]);
        box = bound_union(box, bbox_of_triangle(v0, v1, v2));
    }

    return box;
}

static std::shared_ptr<TriangleMesh> process_mesh(aiMesh* mesh, const aiScene* scene, const MaterialPtr m) {
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

    return std::make_shared<TriangleMesh>(Transform{}, std::move(vs), std::move(idx), m);
}

static void process_node(aiNode* node, const aiScene* scene, std::vector<std::shared_ptr<Hitable>>& meshes, const MaterialPtr m) {
    for (uint i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        meshes.emplace_back(process_mesh(mesh, scene, m));
    }

    for (uint i = 0; i < node->mNumChildren; i++) {
        process_node(node->mChildren[i], scene, meshes, m);
    }
}

std::vector<std::shared_ptr<Hitable>> load_triangle_mesh(const std::string& file_path, const MaterialPtr m) {
    std::vector<std::shared_ptr<Hitable>> meshes;

    if (!fs::exists(fs::absolute(file_path))) {
        std::cout << "file : " << file_path << " does not exist" << std::endl;
        return meshes;
    }

    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(file_path, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_OptimizeMeshes | aiProcess_OptimizeGraph);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        return meshes;
    }

    process_node(scene->mRootNode, scene, meshes, m);

    return meshes;
}