#pragma once

#include <array>

template <typename T, unsigned int N>
class Vec {
public:
    Vec() {
        for (int i = 0; i < N; i++)
            arr[i] = T(0);
    }

    template <typename ...Ts>
    Vec(Ts... args) {
        static_assert(sizeof...(Ts) == N);
        arr = { static_cast<T>(args)... };
    }

    inline T& x() {
        return arr[0];
    }

    inline T& y() {
        return arr[1];
    }

    inline T& z() {
        static_assert(N > 2);
        return arr[2];
    }

    inline T& w() {
        static_assert(N > 3);
        return arr[3];
    }

    auto operator +(const Vec& rhs) const {
        Vec tmp;
        for (int i = 0; i < N; i++)
            tmp.arr[i] = arr[i] + rhs.arr[i];
        return tmp;
    }

    auto& operator +=(const Vec& rhs) {
        for (int i = 0; i < N; i++)
            arr[i] += rhs.arr[i];
        return *this;
    }

    auto operator -(const Vec& rhs) const {
        Vec tmp;
        for (int i = 0; i < N; i++)
            tmp.arr[i] = arr[i] - rhs.arr[i];
        return tmp;
    }

    auto& operator -=(const Vec& rhs) {
        for (int i = 0; i < N; i++)
            arr[i] -= rhs.arr[i];
        return *this;
    }

    auto operator *(const float s) const {
        Vec tmp;
        for (int i = 0; i < N; i++)
            tmp.arr[i] = arr[i] * s;
        return tmp;
    }

    auto operator *=(const float s) const {
        for (int i = 0; i < N; i++)
            arr[i] *= s;
        return *this;
    }

    friend auto operator *(const float s, const Vec& rhs) {
        return rhs * s;
    }

    T dot(const Vec& rhs) const {
        T tmp{0};
        for (int i = 0; i < N; i++)
            tmp += arr[i] * rhs.arr[i];
        return tmp;
    }

    auto cross(const Vec& rhs) const {
        static_assert(N > 1 && N < 4);
        if constexpr (N == 2) {
            return arr[0] * rhs.arr[1] - arr[1] * rhs.arr[0];
        }
        else {
            Vec tmp;
            tmp[0] = arr[1] * rhs.arr[2] - arr[2] * rhs.arr[1];
            tmp[1] = arr[2] * rhs.arr[0] - arr[0] * rhs.arr[2];
            tmp[2] = arr[0] * rhs.arr[1] - arr[1] * rhs.arr[0];
        }
    }

protected:
    std::array<T, N> arr;
};

template <typename T>
using Vec2 = Vec<T, 2>;

template <typename T>
using Vec3 = Vec<T, 3>;

template <typename T>
using Vec4 = Vec<T, 4>;

using Vec2f = Vec2<float>;
using Vec2d = Vec2<double>;
using Vec3f = Vec3<float>;
using Vec3d = Vec3<double>;
using Vec4f = Vec4<float>;
using Vec4d = Vec4<double>;
