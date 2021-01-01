#pragma once

#include "base/mat.h"
#include "base/utils.h"

template <typename T>
class Transform {
public:

    using ValueType = T;

    Transform()
        : mat(Mat4<T>::identity())
        , mat_inv(Mat4<T>::identity())
    {}

    Transform(const Mat4<T>& m)
        : mat(m) {
        mat_inv = m.inverse();
    }

    Transform(Mat4<T>&& m)
        : mat(m) {
        mat_inv = m.inverse();
    }

    Transform(const Mat4<T>& m, const Mat4<T>& inv)
        : mat(m)
        , mat_inv(inv)
    {}

    inline Transform inverse() const {
        return Transform(mat_inv, mat);
    }

    Transform& translate(const Vec3<T>& t) {
        for (int i = 0; i < t.dimension; i++) {
            mat[i][3] += t[i];
            mat_inv[i][3] -= t[i];
        }

        return *this;
    }

    //void translate(const T& x, const T& y, const T& z);

    Transform& rotate(const Vec3<T>& axis, const T& angle) {
        auto a = axis.normalized();
        auto angle_in_radian = to_radian<T>(angle);
        T sin_theta = std::sin(angle_in_radian);
        T cos_theta = std::cos(angle_in_radian);

        // Compute rotation of first basis vector
        Mat4<T> rot;
        rot[0][0] = a.x() * a.x() + (1. - a.x() * a.x()) * cos_theta;
        rot[1][0] = a.x() * a.y() * (1. - cos_theta) - a.z() * sin_theta;
        rot[2][0] = a.x() * a.z() * (1. - cos_theta) + a.y() * sin_theta;
        rot[3][0] = 0.;

        rot[0][1] = a.x() * a.y() * (1. - cos_theta) + a.z() * sin_theta;
        rot[1][1] = a.y() * a.y() + (1. - a.y() * a.y()) * cos_theta;
        rot[2][1] = a.y() * a.z() * (1. - cos_theta) - a.x() * sin_theta;
        rot[3][1] = 0.;

        rot[0][2] = a.x() * a.z() * (1. - cos_theta) - a.y() * sin_theta;
        rot[1][2] = a.y() * a.z() * (1. - cos_theta) + a.x() * sin_theta;
        rot[2][2] = a.z() * a.z() + (1. - a.z() * a.z()) * cos_theta;
        rot[2][3] = 0;

        // Column majored matrix, apply transform in the left
        mat = rot * mat;
        auto rot_inv = rot.transpose();
        mat_inv = rot_inv * mat_inv;

        return *this;
    }

    Transform& scale(const Vec3<T>& s) {
        for (int i = 0; i < s.dimension; i++) {
            mat[i][i] *= s[i];
            mat_inv[i][i] /= s[i];
        }
        
        return *this;
    }

private:
    Mat4<T> mat = Mat4<T>::identity();
    Mat4<T> mat_inv = Mat4<T>::identity();
};

using Transformf = Transform<float>;
using Transformd = Transform<double>;