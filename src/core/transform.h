#pragma once

#include "base/mat.h"
#include "base/utils.h"
#include "ray.h"
#include "material.h"

using base::Vec4f;
using base::Mat4f;

class Transform {
public:
    Transform() {}

    Transform(const Mat4f& m)
        : mat(m) {
        mat_inv = base::inverse(m);
    }

    Transform(Mat4f&& m)
        : mat(m) {
        mat_inv = base::inverse(m);
    }

    Transform(const Mat4f& m, const Mat4f& inv)
        : mat(m)
        , mat_inv(inv)
    {}

    inline Transform inverse() const {
        return Transform(mat_inv, mat);
    }

    inline Transform& translate(const Vec3f& t) {
        mat *= base::translate3f(t);
        mat_inv *= base::translate3f(-t);
        return *this;
    }

    Transform& rotate(const Vec3f& axis, const float& radian) {
        // Use radian directly since DCC exports radian by default
        mat *= base::rotate3f(axis, radian);
        mat_inv *= base::rotate3f(axis, -radian);
        return *this;
    }

    inline Transform& scale(const Vec3f& s) {
        mat *= base::scale3f(s);
        mat_inv *= base::scale3f(1.f / s);
        return *this;
    }

    inline Vec3f apply(const Vec3f& v, bool is_vector=false) const {
        //return (mat * Vec4f(v, is_vector ? 0.f : 1.f)).reduct<3>();
        return base::head<3>((mat * concat(v, is_vector ? 0.f : 1.f)));
    }

    /*
    inline Vec4f apply(const Vec4f& v) const {
        return mat * v;
    }
    */

    inline Vec3f apply_normal(const Vec3f& v) const {
        return base::head<3>(transpose(mat_inv) * concat(v, 0.f));
    }

    Intersection apply(const Intersection& isect) const {
        Intersection ret;
        ret.P = apply(isect.P);
        ret.N = base::head<3>((transpose(mat_inv) * concat(isect.N, 0.f)));
        ret.shading_normal = isect.shading_normal;
        ret.tangent = apply(isect.tangent, true);
        ret.bitangent = apply(isect.bitangent, true);
        ret.bary = isect.bary;
        ret.uv = isect.uv;
        ret.ray_t = isect.ray_t;
        ret.backface = isect.backface;
        ret.is_light = isect.is_light;
        ret.geom_id = isect.geom_id;
        ret.shader_name = isect.shader_name;
        ret.light_id = isect.light_id;

        return ret;
    }

    Ray apply(const Ray& r) const {
        auto origin_t = apply(r.origin);
        auto direction_t = apply(r.direction, true);
        return Ray{origin_t, direction_t, r.time, r.tmin, r.tmax};
    }

    friend std::ostream& operator <<(std::ostream& os, const Transform& t) {
        os << "Transform :\n" << t.mat << std::endl;
        return os;
    }

public:
    Mat4f mat = base::identity<float, 4>();
    Mat4f mat_inv = base::identity<float, 4>();
};
