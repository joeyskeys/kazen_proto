#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <iostream>

#include "types.h"

template <typename T, unsigned int N>
class Vec {
public:

    using ValueType = T;

    static constexpr uint dimension = N;

    Vec() {
        //for (int i = 0; i < N; i++)
            //arr[i] = T(0);
        std::fill(arr.begin(), arr.end(), static_cast<T>(0));
    }

    template <typename ...Ts>
    Vec(Ts... args) {
        static_assert(sizeof...(Ts) == N, "dimensional error");
        arr = { static_cast<T>(args)... };
    }

    inline T& x() {
        return arr[0];
    }

    inline T& y() {
        return arr[1];
    }

    inline T& z() {
        static_assert(N > 2, "This vec does not have z component");
        return arr[2];
    }

    inline T& w() {
        static_assert(N > 3, "This vec does not have w component");
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

    auto operator *(const T s) const {
        Vec tmp;
        for (int i = 0; i < N; i++)
            tmp.arr[i] = arr[i] * s;
        return tmp;
    }

    auto operator *=(const T s) const {
        for (int i = 0; i < N; i++)
            arr[i] *= s;
        return *this;
    }

    friend auto operator *(const T s, const Vec& rhs) {
        return rhs * s;
    }

    auto operator /(const T s) const {
        Vec tmp = *this;
        auto inv = static_cast<T>(1) / s;
        for (int i = 0; i < N; i++)
            tmp.arr[i] *= inv;
        return tmp;
    }
    
    auto operator /=(const T s) {
        auto inv = static_cast<T>(1);
        for (int i = 0; i < N; i++)
            arr[i] *= inv;
        return *this;
    }

    bool operator ==(const Vec& rhs) {
        bool eq = true;
        for (int i = 0; i < N; i++)
            eq &= arr[i] == rhs.arr[i];
        return eq;
    }

    T& operator [](const uint idx) {
        assert(idx < N);
        return arr[idx];
    }

    const T& operator [](const uint idx) const {
        assert(idx < N);
        return arr[idx];
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
            return tmp;
        }
    }

    void normalize() {
        T sum{0};
        for (auto& e : arr)
            sum += e * e;
        T rcp = 1. / std::sqrt(static_cast<double>(sum));
        for (auto& e : arr)
            e *= rcp;
    }

    auto normalized() const {
        T sum{0};
        for (auto& e : arr)
            sum += e * e;
        T rcp = 1. / std::sqrt(static_cast<double>(sum));
        return *this * rcp;
    }

    auto length_squared() const {
        T sum{0};
        for (auto& e : arr)
            sum += e * e;
        return sum;
    }

    inline auto length() const {
        return sqrt(length_squared());
    }

    friend std::ostream& operator <<(std::ostream& os, const Vec& v) {
        for (auto& e : v.arr)
            std::cout << e << " ";
        std::cout << std::endl;

        return os;
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
