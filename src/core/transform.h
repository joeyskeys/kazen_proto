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
        /*
        for (int i = 0; i < t.dimension; i++) {
            mat[3][i] += t[i];
            mat_inv[3][i] -= t[i];
        }
        */

        mat *= base::translate3f(t);
        mat_inv *= base::translate3f(-t);
        return *this;
    }

    //void translate(const T& x, const T& y, const T& z);

    Transform& rotate(const Vec3f& axis, const float& angle) {
        /*
        auto a = base::normalize(axis);
        auto angle_in_radian = to_radian<float>(angle);
        auto sin_theta = std::sin(angle_in_radian);
        auto cos_theta = std::cos(angle_in_radian);

        // Compute rotation of first basis vector
        Mat4f rot;
        rot[0][0] = a.x() * a.x() + (1.f - a.x() * a.x()) * cos_theta;
        rot[1][0] = a.x() * a.y() * (1.f - cos_theta) - a.z() * sin_theta;
        rot[2][0] = a.x() * a.z() * (1.f - cos_theta) + a.y() * sin_theta;
        rot[3][0] = 0.f;

        rot[0][1] = a.x() * a.y() * (1.f - cos_theta) + a.z() * sin_theta;
        rot[1][1] = a.y() * a.y() + (1.f - a.y() * a.y()) * cos_theta;
        rot[2][1] = a.y() * a.z() * (1.f - cos_theta) - a.x() * sin_theta;
        rot[3][1] = 0.f;

        rot[0][2] = a.x() * a.z() * (1.f - cos_theta) - a.y() * sin_theta;
        rot[1][2] = a.y() * a.z() * (1.f - cos_theta) + a.x() * sin_theta;
        rot[2][2] = a.z() * a.z() + (1.f - a.z() * a.z()) * cos_theta;
        rot[2][3] = 0.f;

        // Column majored matrix, apply transform in the left
        mat = rot * mat;
        auto rot_inv = base::transpose(rot);
        mat_inv = rot_inv * mat_inv;

        return *this;
        */

        mat *= base::rotate3f(axis, angle);
        mat_inv *= base::rotate3f(axis, -angle);
        return *this;
    }

    inline Transform& scale(const Vec3f& s) {
        /*
        for (int i = 0; i < s.dimension; i++) {
            mat[i][i] *= s[i];
            mat_inv[i][i] /= s[i];
        }
        */
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
