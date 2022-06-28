#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <iostream>
#include <numeric>
#include <type_traits>

#include <OSL/oslconfig.h>

#include "base/basic_types.h"
#include "config.h"

namespace base
{

#ifdef USE_ENOKI

// The reason why not using inheritance to have the same OOP interface is
// because if you don't reimplement all the operator, the return enoki::Array
// object cannot be converted to the inherited class object.

template <typename T, size_t N>
using Vec = enoki::Array<T, N>;

// Construct funcs
template <typename T, size_t N>
inline Vec<T, N + 1> concat(const Vec<T, N>& v, const typename Vec<T, N>::Scalar& s) {
    return enoki::concat(v, s);
}

template <typename T, size_t N, size_t M>
inline Vec<T, N + M> concat(const Vec<T, N>& v1, const Vec<T, M>& v2) {
    return enoki::concat(v1, v2);
}

template <size_t M, typename T, size_t N>
inline Vec<T, M> head(const Vec<T, N>& v) {
    static_assert(N > M);
    return enoki::head<M>(v);
}

// Horizontal funcs
template <typename T, size_t N>
inline T max_component(const Vec<T, N>& v) {
    return enoki::hmax(v);
}

template <typename T, size_t N>
inline T min_component(const Vec<T, N>& v) {
    return enoki::hmin(v);
}

template <typename T, size_t N>
inline T sum(const Vec<T, N>& v) {
    return enoki::hsum(v);
}

template <typename T, size_t N>
inline T length(const Vec<T, N>& v) {
    return enoki::norm(v);
}

template <typename T, size_t N>
inline T length_squared(const Vec<T, N>& v) {
    return enoki::squared_norm(v);
}

// Dot & cross
template <typename T, size_t N>
inline T dot(const Vec<T, N>& v1, const Vec<T, N>& v2) {
    return enoki::dot(v1, v2);
}

template <typename T, size_t N>
inline Vec<T, N> cross(const Vec<T, N>& v1, const Vec<T, N>& v2) {
    return enoki::cross(v1, v2);
}

// Normalize
template <typename T, size_t N>
inline Vec<T, N> normalize(const Vec<T, N>& v) {
    return enoki::normalize(v);
}

// Misc
template <typename T, size_t N>
inline Vec<T, N> abs(const Vec<T, N>& v) {
    return enoki::abs(v);
}

// Comparison
template <typename T, size_t N>
inline bool is_zero(const Vec<T, N>& v, const T& epsilon=1e-5) {
    return enoki::all(abs(v) < epsilon);
}

template <typename T, size_t N>
inline Vec<T, N> vec_min(const Vec<T, N>& a, const Vec<T, N>& b) {
    return enoki::min(a, b);
}

template <typename T, size_t N>
inline Vec<T, N> vec_max(const Vec<T, N>& a, const Vec<T, N>& b) {
    return enoki::max(a, b);
}

// Type conversion
template <typename T, size_t N>
OSL::Vec2 to_osl_vec2(const Vec<T, N>& v) {
    OSL::Vec2 ret;
    ret.x = v[0];
    ret.y = v[1];
    return ret;
}

template <typename T, size_t N>
OSL::Vec3 to_osl_vec3(const Vec<T, N>& v) {
    static_assert(N > 2);
    OSL::Vec3 ret;
    ret.x = v[0];
    ret.y = v[1];
    ret.z = v[2];
    return ret;
}

// stringify
template <typename T, size_t N>
std::string to_string(const Vec<T, N>& v) {
    std::string ret = std::to_string(v[0]);
    for (int i = 1; i < Vec<T, N>::Size; i++)
        ret += " " + std::to_string(v[i]);
    return ret;
}

#elif defined USE_EIGEN

template <typename T, int N>
class Vec : public Eigen::Matrix<T, N, 1> {
public:
    using Scalar = T;
    using Base = Eigen::Matrix<T, N, 1>;
    static constexpr int Size = N;

    Vec(T v=static_cast<T>(0)) {
        Base::setConstant(v);
    }

    template <typename ...Ts, typename = std::enable_if_t<(... && std::is_arithmetic_v<Ts>)>>
    Vec(Ts... args) : Base(static_cast<T>(args)...) {}

    template <typename Derived>
    Vec(const Eigen::MatrixBase<Derived>& p) : Base(p) {}

    template <typename Derived>
    inline Vec &operator =(const Eigen::MatrixBase<Derived>& p) {
        this->Base::operator=(p);
        return *this;
    }

    inline Vec<T, N> operator *(const T s) const {
        auto ret = *this * s;
        return ret;
    }

    template <typename Derived>
    inline Vec<T, N> operator *(const Eigen::MatrixBase<Derived>& v) const {
        auto ret = this->colwise() * v.array();
        return ret;
    }

    template <typename Derived>
    inline Vec<T, N> operator /=(const Eigen::MatrixBase<Derived>& v) {
        *this = this->colwise() / v;
        return *this;
    }
};

// Checkout this page for information for the signature of following functions:
// https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html#:~:text=Eigen's%20use%20of%20expression%20templates,be%20passed%20to%20the%20function.

// Operators
template <typename S, typename Derived, typename = std::enable_if_t<std::is_arithmetic_v<S>>>
auto operator /(const S s, const Eigen::MatrixBase<Derived>& rhs) {
    return rhs.inverse() * static_cast<typename Eigen::MatrixBase<Derived>::Scalar>(s);
}

/*
template <typename Derived1, typename Derived2>
bool operator ==(const Eigen::MatrixBase<Derived1>& v1, const Eigen::MatrixBase<Derived2>& v2) {
    return (v1 - v2).abs().isZero();
}
*/

// Construct funcs
template <typename Derived>
inline auto concat(const Eigen::MatrixBase<Derived>& v, const typename Eigen::MatrixBase<Derived>::Scalar& s) {
    Eigen::Matrix<typename Eigen::MatrixBase<Derived>::Scalar, Eigen::Dynamic, 1> ret;
    ret.resize(v.rows() + 1, 1);
    ret << v, s;
    return ret;
}

template <typename Derived1, typename Derived2>
inline auto concat(const Eigen::MatrixBase<Derived1>& v1, const Eigen::MatrixBase<Derived2>& v2) {
    Eigen::Matrix<typename Eigen::MatrixBase<Derived1>::Scalar, Eigen::Dynamic, 1> ret;
    ret.resize(v1.rows() + v2.rows(), 1);
    ret << v1, v2;
    return ret;
}

template <int M, typename Derived>
inline auto head(const Eigen::MatrixBase<Derived>& v) {
    return v.head(M);
}

// Horizontal funcs
template <typename T, int N>
inline T max_component(const Vec<T, N>& v) {
    return v.maxCoeff();
}

template <typename T, int N>
inline T min_component(const Vec<T, N>& v) {
    return v.minCoeff();
}

template <typename T, int N>
inline T sum(const Vec<T, N>& v) {
    return v.sum();
}

template <typename Derived>
inline Eigen::MatrixBase<Derived>::Scalar length(const Eigen::MatrixBase<Derived>& v) {
    return v.norm();
}

template <typename Derived>
inline Eigen::MatrixBase<Derived>::Scalar length_squared(const Eigen::MatrixBase<Derived>& v) {
    return v.squaredNorm();
}

// Dot & cross
template <typename Derived>
inline auto dot(const Eigen::MatrixBase<Derived>& v1, const Eigen::MatrixBase<Derived>& v2) {
    return v1.dot(v2);
}

template <typename Derived>
inline auto cross(const Eigen::MatrixBase<Derived>& v1, const Eigen::MatrixBase<Derived>& v2) {
    return v1.cross(v2);
}

// Normalize
template <typename Derived>
inline auto normalize(const Eigen::MatrixBase<Derived>& v) {
    return v.normalized();
}

// Misc
template <typename T, int N>
inline Vec<T, N> abs(const Vec<T, N>& v) {
    return v.cwiseAbs();
}

// Comparison
template <typename T, int N>
inline bool is_zero(const Vec<T, N>& v, const T& epsilon=1e-5) {
    return v.isZero(epsilon);
}

template <typename T, int N>
inline Vec<T, N> vec_min(const Vec<T, N>& a, const Vec<T, N>& b) {
    return a.cwiseMin(b);
}

template <typename T, int N>
inline Vec<T, N> vec_max(const Vec<T, N>& a, const Vec<T, N>& b) {
    return a.cwiseMax(b);
}

// Type conversion
template <typename T, int N>
OSL::Vec2 to_osl_vec2(const Vec<T, N>& v) {
    OSL::Vec2 ret;
    ret.x = v(0, 0);
    ret.y = v(1, 0);
    return ret;
}

template <typename T, int N>
OSL::Vec3 to_osl_vec3(const Vec<T, N>& v) {
    static_assert(N > 2);
    OSL::Vec3 ret;
    ret.x = v(0, 0);
    ret.y = v(1, 0);
    ret.z = v(2, 0);
    return ret;
}

// Stringify
template <typename T, int N>
std::string to_string(const Vec<T, N>& v) {
    std::string ret = std::to_string(v(0, 0));
    for (int i = 0; i < N; i++)
        ret += " " + std::to_string(v(0, i));
    return ret;
}

#else

template <typename T, uint N>
class Vec {
public:

    using Scalar = T;

    static constexpr uint Size = N;

    Vec() {
        std::fill(arr.begin(), arr.end(), static_cast<T>(0));
    }

    template <typename M, typename = std::enable_if_t<std::is_arithmetic_v<M>>>
    Vec(M arg) {
        std::fill(arr.begin(), arr.end(), static_cast<T>(arg));
    }

    template <typename ...Ts, typename = std::enable_if_t<(... && std::is_arithmetic_v<Ts>)>>
    Vec(Ts... args) {
        arr = { static_cast<T>(args)... };
    }

    template <uint M, typename ...Ts>
    Vec(const Vec<T, M>& v, Ts... args) {
        static_assert(M + sizeof...(Ts) == N, "The amount of components differs");
        static_assert(M > 1 && M < N, "The sub vector should have less components");
        std::array<T, N - M> sub = { static_cast<T>(args)... };
        for (int i = 0; i < M; i++)
            arr[i] = v[i];
        constexpr int subcnt = N - M;
        for (int i = 0; i < subcnt; i++)
            arr[M + i] = sub[i];
    }

    Vec(const OSL::Vec2& v) {
        static_assert(N == 2);

        for (int i = 0; i < N; i++)
            arr[i] = v[i];
    }

    Vec(const OSL::Vec3& v) {
        static_assert(N >= 3);

        for (int i = 0; i < N; i++)
            arr[i] = v[i];

        if constexpr (N == 4)
            arr[3] = static_cast<T>(0);
    }

    inline T& x() {
        return arr[0];
    }

    inline const T& x() const {
        return arr[0];
    }

    inline T& y() {
        return arr[1];
    }
    
    inline const T& y() const {
        return arr[1];
    }

    inline T& z() {
        static_assert(N > 2, "This vec does not have z component");
        return arr[2];
    }

    inline const T& z() const {
        static_assert(N > 2, "This vec does not have z component");
        return arr[2];
    }

    inline T& w() {
        static_assert(N > 3, "This vec does not have w component");
        return arr[3];
    }

    inline const T& w() const {
        static_assert(N > 3, "This vec does not have w component");
        return arr[3];
    }

    inline T& r() {
        return arr[0];
    }

    inline const T& r() const {
        return arr[0];
    }

    inline T& g() {
        return arr[1];
    }

    inline const T& g() const {
        return arr[1];
    }

    inline T& b() {
        static_assert(N > 2, "This vec does not have b component");
        return arr[2];
    }

    inline const T& b() const {
        static_assert(N> 2, "This vec does not have b component");
    }

    inline T* data() {
        return arr.data();
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

    auto operator -() const {
        Vec tmp;
        for (int i = 0; i < N; i++)
            tmp.arr[i] = -arr[i];
        return tmp;
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

    auto operator *(const Vec& rhs) const {
        Vec tmp;
        for (int i = 0; i < N; i++)
            tmp.arr[i] = arr[i] * rhs.arr[i];
        return tmp;
    }

    auto operator *=(const T s) {
        for (int i = 0; i < N; i++)
            arr[i] *= s;
        return *this;
    }

    auto operator *=(const Vec& rhs) {
        for (int i = 0; i < N; i++)
            arr[i] *= rhs.arr[i];
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

    auto operator /(const Vec<T, N>& v) const {
        Vec tmp = *this;
        for (int i = 0; i < N; i++)
            tmp[i] = arr[i] / v[i];
        return tmp;
    }
    
    auto operator /=(const T s) {
        auto inv = static_cast<T>(1) / s;
        for (int i = 0; i < N; i++)
            arr[i] *= inv;
        return *this;
    }

    auto operator /=(const Vec<T, N>& v) {
        for (int i = 0; i < N; i++) 
            arr[i] /= v[i];
        return *this;
    }

    friend auto operator /(const T s, const Vec& rhs) {
        Vec tmp;
        for (int i = 0; i < N; i++)
            tmp.arr[i] = s / rhs.arr[i];
        return tmp;
    }

    bool operator ==(const Vec& rhs) const {
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

    template <uint M>
    Vec<T, M> head() const {
        static_assert(M > 1 && M < N, "Invalid component number");
        Vec<T, M> tmp;
        for (int i = 0; i < M; i++)
            tmp[i] = arr[i];

        return tmp;
    }

    operator OSL::Vec2() const {
        OSL::Vec2 ret;
        ret.x = arr[0];
        ret.y = arr[1];
        return ret;
    }

    operator OSL::Vec3() const {
        static_assert(N > 2);
        OSL::Vec3 ret;
        ret.x = arr[0];
        ret.y = arr[1];
        ret.z = arr[2];
        return ret;
    }

    template <typename C, typename = std::enable_if_t<std::is_base_of_v<Vec, C>>>
    T dot(const C& rhs) const {
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

    bool is_zero() const {
        for (int i = 0; i < N; i++)
            if (std::abs(arr[i]) >= epsilon<T>)
                return false;
        return true;
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

    inline auto sum() const {
        // Be very careful when using template function
        // Using 0 for third parameter directly cause inf...
        return std::accumulate(arr.begin(), arr.end(), T{0});
    }

    inline auto abs() const {
        Vec tmp;
        for (int i = 0; i < N; i++)
            tmp.arr[i] = std::abs(arr[i]);
        return tmp;
    }

    inline auto max_component() const {
        return *std::max_element(arr.begin(), arr.end());
    }

    inline auto min_component() const {
        return *std::min_element(arr.begin(), arr.end());
    }

    std::string to_str() const {
        std::string ret = std::to_string(arr[0]);
        for (int i = 1; i < N; i++)
            ret += " " + std::to_string(arr[i]);
        return ret;
    }

    friend std::ostream& operator <<(std::ostream& os, const Vec& v) {
        for (auto& e : v.arr)
            os << e << " ";

        return os;
    }

    friend std::stringstream& operator >>(std::stringstream& ss, Vec& v) {
        for (auto& e : v.arr)
            ss >> e;
        
        return ss;
    }

public:
    std::array<T, N> arr;
};

// Construct funcs
template <typename T, uint N>
inline Vec<T, N + 1> concat(const Vec<T, N>& v, const typename Vec<T, N>::Scalar& s) {
    return Vec<T, N + 1>{v, s};
}

template <typename T, uint N, uint M>
inline Vec<T, N + M> concat(const Vec<T, N>& v1, const Vec<T, M>& v2) {
    return Vec<T, N + M>{v1, v2};
}

template <uint M, typename T, uint N>
inline Vec<T, M> head(const Vec<T, N>& v) {
    /*
    static_assert(N > M && M > 1, "Invalid component number");
    Vec<T, M> tmp;
    for (int i = 0; i < M; i++)
        tmp[i] = v.arr[i];

    return tmp;
    */
    return v.template head<M>();
}

// Horizontal funcs
template <typename T, uint N>
inline T max_component(const Vec<T, N>& v) {
    //return *std::max_element(arr.begin(), arr.end());
    return v.max_component();
}

template <typename T, uint N>
inline T min_component(const Vec<T, N>& v) {
    //return *std::min_element(arr.begin(), arr.end());
    return v.min_component();
}

template <typename T, uint N>
inline T sum(const Vec<T, N>& v) {
    //return std::accumulate(arr.begin(), arr.end(), T{0});
    return v.sum();
}

template <typename T, uint N>
inline T length(const Vec<T, N>& v) {
    return v.length();
}

template <typename T, uint N>
inline T length_squared(const Vec<T, N>& v) {
    return v.length_squared();
}

// Dot & cross
template <template<typename, uint> class C, typename T, uint N, typename = std::enable_if_t<std::is_base_of_v<Vec<T, N>, C<T, N>>>>
inline T dot(const C<T, N>& a, const C<T, N>& b) {
    return a.dot(b);
    /*
    T tmp{0};
    for (int i = 0; i < N; i++)
        tmp += a.arr[i] * b.arr[i];
    return tmp;
    */
}

template <template<typename, uint> class C, typename D, typename T, uint N,
    typename = std::enable_if_t<std::is_base_of_v<Vec<T, N>, C<T, N>>>, typename = std::enable_if_t<std::is_convertible_v<D, C<T, N>>>>
inline T dot(const C<T, N>& a, const D& b) {
    //return a.dot(static_cast<C<T, N>>(b));
    return dot(a, static_cast<C<T, N>>(b));
}

template <typename T, uint N>
inline auto cross(const Vec<T, N>& a, const Vec<T, N>& b) {
    return a.cross(b);
    /*
    static_assert(N > 1 && N < 4);
    if constexpr (N == 2) {
        return a.arr[0] * b.arr[1] - a.arr[1] * b.arr[0];
    }
    else {
        Vec tmp;
        tmp[0] = a.arr[1] * b.arr[2] - a.arr[2] * b.arr[1];
        tmp[1] = a.arr[2] * b.arr[0] - a.arr[0] * b.arr[2];
        tmp[2] = a.arr[0] * b.arr[1] - a.arr[1] * b.arr[0];
        return tmp;
    }
    */
}

// Normalize
template <typename T, uint N>
inline Vec<T, N> normalize(const Vec<T, N>& w) {
    return w.normalized();
    /*
    T sum{0};
    for (auto& e : arr)
        sum += e * e;
    T rcp = 1. / std::sqrt(static_cast<double>(sum));
    for (auto& e : arr)
        e *= rcp;
    */
}

// Misc
template <typename T, uint N>
inline Vec<T, N> abs(const Vec<T, N>& v) {
    return v.abs();
}

// Comparison
template <typename T, uint N>
inline bool is_zero(const Vec<T, N>& v) {
    /*
    for (int i = 0; i < N; i++)
        if (std::abs(v.arr[i]) >= epsilon<T>)
            return false;
    return true;
    */
    return v.is_zero();
}

template <typename T, uint N>
inline Vec<T, N> vec_min(const Vec<T, N>& a, const Vec<T, N>& b) {
    Vec<T, N> tmp;
    for (int i = 0; i < N; i++)
        tmp[i] = std::min(a[i], b[i]);
    return tmp;
}

template <typename T, uint N>
inline Vec<T, N> vec_max(const Vec<T, N>& a, const Vec<T, N>& b) {
    Vec<T, N> tmp;
    for (int i = 0; i < N; i++)
        tmp[i] = std::max(a[i], b[i]);
    return tmp;
}

// Type conversion
template <typename T, uint N>
OSL::Vec2 to_osl_vec2(const Vec<T, N>& v) {
    OSL::Vec2 ret;
    ret.x = v[0];
    ret.y = v[1];
    return ret;
}

template <typename T, uint N>
OSL::Vec3 to_osl_vec3(const Vec<T, N>& v) {
    static_assert(N > 2);
    OSL::Vec3 ret;
    ret.x = v[0];
    ret.y = v[1];
    ret.z = v[2];
    return ret;
}

// Stringify
template <typename T, uint N>
inline std::string to_string(const Vec<T, N>& v) {
    return v.to_str();
}

#endif

template <typename T>
using Vec2 = Vec<T, 2>;

template <typename T>
using Vec3 = Vec<T, 3>;

template <typename T>
using Vec4 = Vec<T, 4>;

using Vec2f = Vec2<float>;
using Vec2d = Vec2<double>;
using Vec2i = Vec2<int>;
using Vec3f = Vec3<float>;
using Vec3d = Vec3<double>;
using Vec3i = Vec3<int>;
using Vec4f = Vec4<float>;
using Vec4d = Vec4<double>;
using Vec4i = Vec4<int>;

inline Vec2f to_vec2(const OSL::Vec2& v) {
    return Vec2f{v.x, v.y};
}

inline Vec3f to_vec3(const OSL::Vec3& v) {
    return Vec3f{v.x, v.y, v.z};
}

} // namspace base