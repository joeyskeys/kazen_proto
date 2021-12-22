#pragma once

#include "base/mat.h"
#include "base/utils.h"
#include "ray.h"
#include "material.h"

class Transform {
public:
    Transform()
        : mat(Mat4f::identity())
        , mat_inv(Mat4f::identity())
    {}

    Transform(const Mat4f& m)
        : mat(m) {
        mat_inv = m.inverse();
    }

    Transform(Mat4f&& m)
        : mat(m) {
        mat_inv = m.inverse();
    }

    Transform(const Mat4f& m, const Mat4f& inv)
        : mat(m)
        , mat_inv(inv)
    {}

    inline Transform inverse() const {
        return Transform(mat_inv, mat);
    }

    inline Transform& translate(const Vec3f& t) {
        for (int i = 0; i < t.dimension; i++) {
            mat[3][i] += t[i];
            mat_inv[3][i] -= t[i];
        }

        return *this;
    }

    //void translate(const T& x, const T& y, const T& z);

    Transform& rotate(const Vec3f& axis, const float& angle) {
        auto a = axis.normalized();
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
        auto rot_inv = rot.transpose();
        mat_inv = rot_inv * mat_inv;

        return *this;
    }

    inline Transform& scale(const Vec3f& s) {
        for (int i = 0; i < s.dimension; i++) {
            mat[i][i] *= s[i];
            mat_inv[i][i] /= s[i];
        }
        
        return *this;
    }

    inline Vec3f apply(const Vec3f& v, bool is_vector=false) const {
        return (mat * Vec4f(v, is_vector ? 0.f : 1.f)).reduct<3>();
    }

    inline Vec4f apply(const Vec4f& v) const {
        return mat * v;
    }

    Intersection apply(const Intersection& isect) const {
        Intersection ret;
        ret.position = apply(isect.position);
        ret.normal = (mat_inv.transpose() * Vec4f(isect.normal, 0.f)).reduct<3>();
        ret.shading_normal = isect.shading_normal;
        ret.tangent = apply(isect.tangent, true);
        ret.bitangent = apply(isect.bitangent, true);
        ret.bary = isect.bary;
        ret.uv = isect.uv;
        ret.ray_t = isect.ray_t;
        ret.backface = isect.backface;
        ret.is_light = isect.is_light;
        ret.obj_id = isect.obj_id;
        ret.shader_name = isect.shader_name;

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
    Mat4f mat = Mat4f::identity();
    Mat4f mat_inv = Mat4f::identity();
};
