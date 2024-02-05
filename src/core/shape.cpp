#include <cmath>
#include <filesystem>
#include <functional>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
//#include <boost/filesystem.hpp>
#include <boost/math/constants/constants.hpp>
#include <fmt/core.h>

#include "core/light.h"
#include "core/ray.h"
#include "core/sampling.h"
#include "core/shape.h"
#include "core/transform.h"

//namespace fs = boost::filesystem;
namespace fs = std::filesystem;
namespace constants = boost::math::constants;

static inline AABBf bbox_of_triangle(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2) {
    return bound_union(AABBf{base::vec_min(v0, v1), base::vec_max(v0, v1)}, v2);
}

void Shape::post_hit(Intersection& isect) const {
    isect.calculate_differentials();
}

void Shape::print_bound() const {
    std::cout << "id : " << geom_id << std::endl;
    Hitable::print_bound();
}

bool Sphere::intersect(const Ray& r, Intersection& isect) const {
    auto r_local = world_to_local.apply(r);

    auto oc = r_local.origin - base::head<3>(center_n_radius);
    auto a = base::length_squared(r_local.direction);
    auto half_b = dot(oc, r_local.direction);
    auto c =base::length_squared(oc) - center_n_radius.w() * center_n_radius.w();
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

    // Return false if not nearer than previous t
    if (t < 0 || t > isect.ray_t)
        return false;

    // Calculate in local space
    isect.ray_t = t;
    // small bias to avoid self intersection
    isect.P = r_local.at(t) * 1.00001f;
    isect.N = (r_local.at(t) - base::head<3>(center_n_radius)) / center_n_radius.w();
    if (isect.backface)
        isect.N = -isect.N;

    if (isect.N == Vec3f(0.f, 1.f, 0.f))
        isect.tangent = base::normalize(base::cross(Vec3f{0.f, 0.f, -1.f}, isect.N));
    else
        isect.tangent = base::normalize(base::cross(Vec3f{0.f, 1.f, 0.f}, isect.N));

    isect.bitangent = base::normalize(base::cross(isect.tangent, isect.N));
    isect.is_light = is_light;
    isect.shape = const_cast<Sphere*>(this);
    isect.geom_id = geom_id;

    float phi = std::atan2(isect.P.z(), isect.P.x());
    if (phi < 0)
        phi += boost::math::constants::two_pi<float>();
    isect.uv[0] = phi * constants::one_div_two_pi<float>();
    float theta = std::acos(std::clamp(isect.P.y() / center_n_radius.w(), -1.f, 1.f));
    isect.uv[1] = theta / constants::pi<float>() + 0.5;

    // Transform back to world space
    isect = local_to_world.apply(isect);
    isect.shader_name = shader_name;

    if (is_light)
        isect.light_id = light->light_id;
    
    return true;
}

bool Sphere::intersect(const Ray& r, float& t) const {
    auto r_local = world_to_local.apply(r);

    auto oc = r_local.origin - base::head<3>(center_n_radius);
    auto a = base::length_squared(r_local.direction);
    auto half_b = base::dot(oc, r_local.direction);
    auto c = base::length_squared(oc) - center_n_radius.w() * center_n_radius.w();
    auto discriminant = half_b * half_b - a * c;

    if (discriminant < 0.f) return false;

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
    auto center_in_world = local_to_world.apply(base::head<3>(center_n_radius));
    auto radius_vec = Vec3f{center_n_radius.w()};
    return AABBf{center_in_world - radius_vec, center_in_world + radius_vec};
}

void* Sphere::address_of(const std::string& n) {
    if (n == "center_n_radius")
        return &center_n_radius;
    else if (n == "name")
        return &name;
    else if (n == "shader_name")
        return &shader_name;
    else if (n == "is_light")
        return &is_light;
    else if (n == "translate")
        // function call param is kinda special
        return this;
    return nullptr;
}

void Sphere::sample(Vec3f& p, Vec3f& n, Vec2f& uv, float& pdf) const {
    auto sample = random3f();

    n = base::normalize(sample);
    n = local_to_world.apply_normal(n);

    p = base::head<3>(center_n_radius) + n * center_n_radius.w();
    p = local_to_world.apply(p);

    uv[1] = acos(center_n_radius.y()) * constants::one_div_pi<float>();
    auto sin_theta_v = asin(center_n_radius.y());
    uv[0] = acos(center_n_radius.x() / sin_theta_v) * constants::one_div_two_pi<float>();

    pdf = 1.f / area();
}

void Sphere::post_hit(Intersection& isect) const {
    // TODO : Calculate the hit position with uv from embree to get more
    // accurate result.
    Shape::post_hit(isect);
    auto up_in_world = local_to_world.apply(Vec3f(0.f, 1.f, 0.f), true);
    auto forward_in_world = local_to_world.apply(Vec3f(0.f, 0.f, -1.f), true);
    if (isect.N == up_in_world)
        isect.tangent = base::normalize(base::cross(forward_in_world, isect.N));
    else
        isect.tangent = base::normalize(base::cross(up_in_world, isect.N));

    isect.bitangent = base::normalize(base::cross(isect.tangent, isect.N));

    isect.is_light = is_light;
    isect.shader_name = shader_name;
    isect.shading_normal = isect.N;

    // Embree do not calculate parametric sphere uv
    // calculate it manually
    auto local_pos = world_to_local.apply(isect.P);
    float phi = std::atan2(local_pos.z(), local_pos.x());
    if (phi < 0)
        phi += constants::two_pi<float>();
    //isect.uv[0] = phi * constants::one_div_two_pi<float>();
    //isect.uv[1] = theta / constants::pi<float>() + 0.5;
    float theta = std::acos(std::clamp(local_pos.y() / center_n_radius.w(), -1.f, 1.f));
    float sinphi, cosphi;
    sincosf(phi, &sinphi, &cosphi);

    isect.dpdu = local_to_world.apply(base::normalize(Vec3f{-constants::two_pi<float>() * local_pos.z(), 0, constants::two_pi<float>() * local_pos.x()}));
    isect.dpdv = local_to_world.apply(base::normalize(Vec3f{local_pos.y() * cosphi, -center_n_radius[3] * std::sin(theta), local_pos.y() * sinphi}));
    //isect.dpdx = isect.tangent;
    //isect.dpdy = isect.bitangent;

    if (is_light)
        isect.light_id = light->light_id;
}

float Sphere::area() const {
    return 4.f * constants::pi<float>() * square(center_n_radius.w());
}

void Sphere::print_info() const {
    std::cout << fmt::format("shape Sphere : radius {}", center_n_radius.w()) << std::endl;
    print_bound();
}

void Sphere::get_world_position(Vec4f* cnr) const {
    auto vec4_cnr = reinterpret_cast<Vec4f*>(cnr);
    *vec4_cnr = base::concat(local_to_world.apply(base::head<3>(center_n_radius)), center_n_radius.w());
}

/*
static inline bool plane_intersect(const Ray& r, const Vec3f& center, const Vec3f& dir, float& t) {
    auto pos_vec = center - r.origin;
    auto projected_distance = base::dot(pos_vec, dir);

    // Back facing hit ignored
    if (projected_distance >= 0)
        return false;

    // Plane direction vector is supposed to be normalized
    // No extra check here
    auto projected_direction = base::dot(dir, r.direction);
    
    // Avoid zero division when ray is parallel to the plane
    if (projected_direction == 0.f)
        return false;

    t = projected_distance / base::dot(dir, r.direction);
    if (t < 0)
        return false;

    return true;
}
*/

static inline bool plane_intersect(const Ray& r, const Vec3f& center, const Vec3f& dir, float& t, Vec3f& pos) {
    plane_intersect(r, center, dir, t);
    pos = r.origin + r.direction * t;
    return true;
}

bool Quad::intersect(const Ray& r, Intersection& isect) const {
    float t;
    Vec3f pos;
    if (!plane_intersect(r, center, dir, t, pos))
        return false;
    if (t < 0 || t > isect.ray_t)
        return false;

    isect.ray_t = t;
    auto position_vec = isect.P - center;
    float horizontal_distance = fabsf(base::dot(position_vec, horizontal_vec));
    float vertical_distance = fabsf(base::dot(position_vec, vertical_vec));

    if (horizontal_distance > half_width ||  vertical_distance > half_height)
        return false;

    isect.N = dir;
    isect.tangent = horizontal_vec;
    isect.bitangent = vertical_vec;
    isect.backface = false;
    isect.is_light = is_light;
    isect.shape = const_cast<Quad*>(this);
    isect.shader_name = shader_name;
    isect.geom_id = geom_id;

    if (is_light)
        isect.light_id = light->light_id;

    return true;
}

bool Quad::intersect(const Ray& r, float& t) const {
    if (!plane_intersect(r, center, dir, t))
        return false;
    return true;
}

AABBf Quad::bbox() const {
    auto down_left_pt = local_to_world.apply(down_left);
    auto down_right_pt = local_to_world.apply(down_left + horizontal_vec * half_width * 2);
    auto up_left_pt = local_to_world.apply(down_left + vertical_vec * half_height * 2);
    auto up_right_pt = local_to_world.apply(down_left + horizontal_vec * half_width * 2 + vertical_vec * half_height * 2);
    auto bbox_of_three_pt = bbox_of_triangle(down_left_pt, down_right_pt, up_left_pt);
    return bound_union(bbox_of_three_pt, up_right_pt);
}

void* Quad::address_of(const std::string& n) {
    if (n == "center")
        return &center;
    else if (n== "dir")
        return &dir;
    else if (n == "vertical_vec")
        return &vertical_vec;
    else if (n == "horizontal_vec")
        return &horizontal_vec;
    else if (n == "half_width")
        return &half_width;
    else if (n == "half_height")
        return &half_height;
    else if (n == "name")
        return &name;
    else if (n == "shader_name")
        return &shader_name;
    else if (n == "is_light")
        return &is_light;
    else if (n == "translate")
        return this;
    else
        return nullptr;
}

void Quad::sample(Vec3f& p, Vec3f& n, Vec2f& uv, float& pdf) const {
    auto sample = random2f();

    p = center + horizontal_vec * half_width * (sample.x() * 2 - 1) + \
        vertical_vec * half_height * (sample.y() * 2 - 1);
    p = local_to_world.apply(p);

    n = dir;
    n = local_to_world.apply_normal(n);

    uv[0] = sample.x();
    uv[1] = sample.y();

    pdf = 1.f / area();
}

void Quad::post_hit(Intersection& isect) const {
    Shape::post_hit(isect);
    isect.tangent = local_to_world.apply(horizontal_vec, true);
    isect.bitangent = local_to_world.apply(vertical_vec, true);
    isect.is_light = is_light;
    isect.shader_name = shader_name;
    isect.shading_normal = isect.N;

    isect.P = center + horizontal_vec * half_width * (isect.uv[0] - 0.5f) +
        vertical_vec * half_height * (isect.uv[1] - 0.5f);

    if (is_light)
        isect.light_id = light->light_id;
}

float Quad::area() const {
    return half_width * half_height * 4.f;
}

void Quad::print_info() const {
    std::cout << fmt::format("shape Quad : center {}, dir {}",
        base::to_string(center), base::to_string(dir)) << std::endl;
    print_bound();
}

void Quad::get_verts(void* vs) const {
    // No bound checking could be dangerous
    auto hor = horizontal_vec * half_width;
    auto ver = vertical_vec * half_height;
    auto left_top = center - hor + ver;
    auto left_bottom = center - hor - ver;
    auto right_top = center + hor + ver;
    auto right_bottom = center + hor - ver;
    auto verts = reinterpret_cast<Vec3f*>(vs);
    verts[0] = right_top;
    verts[1] = right_bottom;
    verts[2] = left_bottom;
    verts[3] = left_top;
}

static bool moller_trumbore_intersect(const Ray& r, const Vec3f* verts, Intersection& isect) {
    Vec3f v1v0 = verts[1] - verts[0];
    Vec3f v2v0 = verts[2] - verts[0];
    Vec3f pvec = base::cross(r.direction, v2v0);
    float det = base::dot(v1v0, pvec);

    if (det < 0.000001)
        return false;

    float det_inv = 1.f / det;
    Vec3f tvec = r.origin - verts[0];
    float u = base::dot(tvec, pvec) * det_inv;
    if (u < 0.f || u > 1.f)
        return false;

    Vec3f qvec = base::cross(tvec, v1v0);
    float v = base::dot(r.direction, qvec) * det_inv; 
    if (v < 0.f || u + v > 1.f)
        return false;

    float t = base::dot(v2v0, qvec) * det_inv;

    // Return false if not nearer than previous hit
    if (t < 0 || t > isect.ray_t)
        return false;

    isect.P = r.at(t - 0.00001f);
    isect.N = base::normalize(base::cross(v1v0, v2v0));
    // TODO : calculate dpdu & dpdv.
    isect.tangent = base::normalize(v1v0);
    isect.bitangent = base::cross(isect.tangent, isect.N);
    isect.ray_t = t;
    isect.backface = t < 0.f ? true : false;

    return true;
}

bool Triangle::intersect(const Ray& r, Intersection& isect) const {
    auto local_r = world_to_local.apply(r);

    auto ret = moller_trumbore_intersect(local_r, verts, isect);
    if (ret && !isect.backface) {
        isect = local_to_world.apply(isect);
        isect.shader_name = shader_name;
        isect.is_light = is_light;
        isect.geom_id = geom_id;
        isect.shape = const_cast<Triangle*>(this);

        if (is_light)
            isect.light_id = light->light_id;

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
    return moller_trumbore_intersect(local_r, verts, t);
}

AABBf Triangle::bbox() const {
    auto v0_in_world = local_to_world.apply(verts[0]);
    auto v1_in_world = local_to_world.apply(verts[1]);
    auto v2_in_world = local_to_world.apply(verts[2]);
    return bbox_of_triangle(v0_in_world, v1_in_world, v2_in_world);
}

void* Triangle::address_of(const std::string& n) {
    if (n== "verta")
        return &verts[0];
    else if (n== "vertb")
        return &verts[1];
    else if (n== "vertc")
        return &verts[2];
    else if (n== "name")
        return &name;
    else if (n== "shader_name")
        return &shader_name;
    else if (n== "is_light")
        return &is_light;
    else if (n== "translate")
        return this;
    return nullptr;
}

void Triangle::sample(Vec3f& p, Vec3f& n, Vec2f& uv, float& pdf) const {
    auto sample = random2f();
    float alpha = 1 - sample.x();
    float beta = alpha * sample.y();

    p = verts[0] + alpha * (verts[1] - verts[0]) + beta * (verts[2] - verts[0]);
    p = local_to_world.apply(p);
    n = local_to_world.apply_normal(normal);

    uv[0] = alpha;
    uv[1] = beta;

    pdf = 1.f / area();
}

void Triangle::post_hit(Intersection& isect) const {
    Shape::post_hit(isect);
    Vec3f v1v0 = verts[1] - verts[0];
    isect.tangent = local_to_world.apply(base::normalize(v1v0), true);
    isect.bitangent = base::cross(isect.tangent, isect.N);
    isect.is_light = is_light;
    isect.shader_name = shader_name;
    isect.shading_normal = isect.N;

    // Compute hit point with barycentric coordinate is more accurate
    // But currently it will cause a weird problem..
    Vec3f bary{1 - isect.uv[0] - isect.uv[1], isect.uv[0], isect.uv[1]};
    isect.P = bary[0] * verts[0] + bary[1] * verts[1] + bary[2] * verts[2];

    if (is_light)
        isect.light_id = light->light_id;
}

float Triangle::area() const {
    return base::length_squared(base::cross(verts[1] - verts[0], verts[2] - verts[0])) * 0.5f;
}

void Triangle::print_info() const {
    std::cout << fmt::format("shape Triangle : verts {} {} {}", verts[0], verts[1], verts[2])
        << std::endl;
    print_bound();
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
            isect.shader_name = shader_name;
            isect.is_light = is_light;
            isect.geom_id = geom_id;
            isect.shape = const_cast<TriangleMesh*>(this);
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

void TriangleMesh::sample(Vec3f& p, Vec3f& n, Vec2f& uv, float& pdf) const {
    //uint idx = randomf() * indice.size();
    auto idx = m_dpdf.sample(randomf());
    
    Vec3f vs[3];
    auto vert_indices = indice[idx];
    vs[0] = verts[vert_indices[0]];
    vs[1] = verts[vert_indices[1]];
    vs[2] = verts[vert_indices[2]];

    auto sp = random2f();
    float su0 = std::sqrt(sp.x());
    float u = 1. - su0;
    float v = sp.y() * su0;
    //p = vs[0] + u * (vs[1] - vs[0]) + v * (vs[2] - vs[0]);
    p = u * vs[0] + v * vs[1] + (1 - u - v) * vs[2];
    p = local_to_world.apply(p);
    n = base::normalize(base::cross(vs[1] - vs[0], vs[2] - vs[0]));
    n = base::normalize(local_to_world.apply_normal(n));
    uv[0] = u;
    uv[1] = v;
    pdf = 1. / m_area;
}

void TriangleMesh::post_hit(Intersection& isect) const {
    auto idx = indice[isect.prim_id];
    auto v1 = verts[idx.x()], v2 = verts[idx.y()], v3 = verts[idx.z()];
    isect.tangent = local_to_world.apply(base::normalize(v2 - v1), true);
    isect.bitangent = base::cross(isect.tangent, isect.N);
    isect.is_light = is_light;
    isect.shader_name = shader_name;
    auto bary_x = 1. - isect.uv[0] - isect.uv[1];
    //isect.P = local_to_world.apply(bary_x * v1 + isect.uv[0] * v2 + isect.uv[1] * v3);
    auto orig_p = bary_x * v1 + isect.uv[0] * v2 + isect.uv[1] * v3;
    // We need P setup for differential calculation
    Shape::post_hit(isect);

    isect.N = local_to_world.apply_normal(isect.N);
    if (norms.size() > 0) {
        auto tnorm0 = local_to_world.apply_normal(norms[idx[0]]);
        auto tnorm1 = local_to_world.apply_normal(norms[idx[1]]);
        auto tnorm2 = local_to_world.apply_normal(norms[idx[2]]);
        isect.shading_normal = base::normalize(bary_x * tnorm0 + isect.uv[0] * tnorm1
            + isect.uv[1] * tnorm2);

        // Apply the method from [1] to solve the shadow terminator
        // problem.
        // [1] HACKING THE SHADOW TERMINATOR
        // https://jo.dreggn.org/home/2021_terminator.pdf
        
        // get distance vectors from triangle vertices
        auto tmpu = orig_p - v1;
        auto tmpv = orig_p - v2;
        auto tmpw = orig_p - v3;
        // project these onto the tangent planes defined by the
        // shading normals
        float dotu = std::min(0.f, base::dot(tmpu, norms[idx[0]]));
        float dotv = std::min(0.f, base::dot(tmpv, norms[idx[1]]));
        float dotw = std::min(0.f, base::dot(tmpw, norms[idx[2]]));
        tmpu -= dotu * tnorm0;
        tmpv -= dotv * tnorm1;
        tmpw -= dotw * tnorm2;
        // Get the new P
        isect.P = local_to_world.apply(orig_p + bary_x * tmpu + isect.uv[0] * tmpv + isect.uv[1] * tmpw);
    }
    else
        isect.shading_normal = isect.N;

    if (uvs.size() > 0) {
        auto uv1 = uvs[idx.x()], uv2 = uvs[idx.y()], uv3 = uvs[idx.z()];
        isect.uv = bary_x * uv1 + isect.uv[0] * uv2 + isect.uv[1] * uv3;
        auto duv13 = uv1 - uv3, duv23 = uv2 - uv3;
        auto dp13 = v1 - v3, dp23 = v2 - v3;
        float determinant = duv13[0] * duv23[1] - duv13[1] * duv23[0];
        bool degenerate = std::abs(determinant) < 1e-8;
        if (!degenerate) {
            float inv = 1 / determinant;
            isect.dpdu = base::normalize((duv23[1] * dp13 - duv13[1] * dp23) * inv);
            isect.dpdv = base::normalize((-duv23[0] * dp13 + duv13[0] * dp23) * inv);
        }
        if (degenerate || base::length_squared(base::cross(isect.dpdu, isect.dpdv)) == 0) {
            isect.dpdu = isect.tangent;
            isect.dpdv = isect.bitangent;
        }
    }
    else {
        isect.dpdu = isect.tangent;
        isect.dpdv = isect.bitangent;
    }

    if (is_light)
        isect.light_id = light->light_id;
}

float TriangleMesh::area() const {
    // Need some pre-calculation;
    return m_area;
}

void TriangleMesh::print_info() const {
    std::cout << "shape TriangleMesh" << std::endl;
}

float TriangleMesh::surface_area(uint32_t i) const {
    auto idxes = indice[i];
    const auto &p0 = verts[idxes[0]], &p1 = verts[idxes[1]], &p2 = verts[idxes[2]];
    return 0.5f * base::length(base::cross(p1 - p0, p2 - p0));
}

void TriangleMesh::setup_dpdf() {
    if (is_light) {
        m_dpdf.reserve(indice.size());
        for (int i = 0; i < indice.size(); ++i) {
            auto area = surface_area(i);
            m_dpdf.append(area);
        }
        m_area = m_dpdf.normalize();
    }
}

static std::shared_ptr<TriangleMesh> process_mesh(aiMesh* mesh, const aiScene* scene, const std::string& m) {
    std::vector<Vec3f> vs(mesh->mNumVertices);
    std::vector<Vec3i> idx(mesh->mNumFaces);

    for (uint i = 0; i < mesh->mNumVertices; i++) {
        vs[i][0] = mesh->mVertices[i].x;
        vs[i][1] = mesh->mVertices[i].y;
        vs[i][2] = mesh->mVertices[i].z;
    }

    std::vector<Vec3f> ns;
    if (mesh->HasNormals()) {
        ns.reserve(mesh->mNumVertices);
        for (uint i = 0; i < mesh->mNumVertices; ++i) {
            ns.emplace_back(Vec3f(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z));
        }
    }

    // Now only checks the first uv set
    std::vector<Vec2f> uvs;
    if (mesh->mTextureCoords[0]) {
        auto uv_cnt = mesh->mNumVertices;
        uvs.resize(mesh->mNumVertices);
        for (uint i = 0; i < uv_cnt; i++) {
            uvs[i] = Vec2f{mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y};
        }
    }

    for (uint i = 0; i < mesh->mNumFaces; i++) {
        idx[i][0] = mesh->mFaces[i].mIndices[0];
        idx[i][1] = mesh->mFaces[i].mIndices[1];
        idx[i][2] = mesh->mFaces[i].mIndices[2];
    }

    return std::make_shared<TriangleMesh>(Transform{}, std::move(vs), std::move(ns), std::move(uvs), std::move(idx), m);
}

static void process_node(aiNode* node, const aiScene* scene, std::vector<std::shared_ptr<TriangleMesh>>& meshes, const std::string& m) {
    for (uint i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        meshes.emplace_back(process_mesh(mesh, scene, m));
    }

    for (uint i = 0; i < node->mNumChildren; i++) {
        process_node(node->mChildren[i], scene, meshes, m);
    }
}

std::vector<std::shared_ptr<TriangleMesh>> load_triangle_mesh(const std::string& file_path, const size_t start_id, const std::string& m) {
    std::vector<std::shared_ptr<TriangleMesh>> meshes;

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
    for (int i = start_id; auto &mesh : meshes) {
        // Fix me: more safe way to do this
        mesh->geom_id = i;
        i += 1;
    }

    return meshes;
}