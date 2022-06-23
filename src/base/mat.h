#pragma once

#include <array>
#include <iostream>
#include <type_traits>

#include <OSL/Imathx/ImathMatrix.h>

#include "base/vec.h"

namespace base
{

#ifdef USE_ENOKI

template <typename T, size_t N>
using Mat = enoki::Matrix<T, N>;

template <typename T>
using Mat2 = enoki::Matrix<T, 2>;

template <typename T>
using Mat3 = enoki::Matrix<T, 3>;

template <typename T>
using Mat4 = enoki::Matrix<T, 4>;

using Mat2f = Mat2<float>;
using Mat2d = Mat2<double>;
using Mat3f = Mat3<float>;
using Mat3d = Mat3<double>;
using Mat4f = Mat4<float>;
using Mat4d = Mat4<double>;

template <typename T, size_t N>
inline Mat<T, N> identity() {
    return enoki::identity<Mat<T, N>>(N);
}

template <typename T, size_t N>
inline Mat<T, N> transpose(const Mat<T, N>& m) {
    return enoki::transpose(m);
}

template <typename T, size_t N>
inline Mat<T, N> inverse(const Mat<T, N>& m) {
    return enoki::inverse(m);
}

template <typename T, size_t N>
inline Mat<T, N> translate(const Vec<T, N - 1>& v) {
    return enoki::translate<Mat<T, N>>(v);
}

const auto translate3f = translate<float, 4>;

template <typename T, size_t N>
inline Mat<T, N> rotate(const Vec<T, N - 1>& axis, const T& angle) {
    return enoki::rotate<Mat<T, N>>(axis, enoki::deg_to_rad(angle));
}

const auto rotate3f = rotate<float, 4>;

template <typename T, size_t N>
inline Mat<T, N> scale(const Vec<T, N - 1>& v) {
    return enoki::scale<Mat<T, N>>(v);
}

const auto scale3f = scale<float, 4>;

template <typename T, size_t N>
OSL::Matrix33 to_osl_mat3(const Mat<T, N>& m) {
    OSL::Matrix33 ret;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            ret[i][j] = m[j][i];
    return ret;
}

template <typename T>
OSL::Matrix44 to_osl_mat4(const Mat4<T>& m) {
    OSL::Matrix44 ret;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            ret[i][j] = m[j][i];
    return ret;
}

#else

template <typename T, unsigned int N>
class Column {
public:
    template <typename Ts>
    Column(Ts* d)
        : data(const_cast<T*>(d))
    {}

    T& operator [](unsigned int idx) {
        // No bound checking for now
        return data[idx];
    }

    const T& operator [](uint idx) const {
        return data[idx];
    }

    void operator +=(const Vec<T, N>& v) {
        for (int i = 0; i < N; i++)
            data[i] += v[i];
    }

    void operator -=(const Vec<T, N>& v) {
        for (int i = 0; i < N; i++)
            data[i] -= v[i];
    }

    bool operator ==(const Vec<T, N>& v) const {
        bool ret = true;
        for (int i = 0; i < N; i++)
            ret &= std::abs(v[i] - data[i]) < epsilon<T>;
        return ret;
    }

private:
    T* data;
};

template <typename T, unsigned int N>
class Mat {
public:

    using ValueType = T;

    static constexpr uint dimension = N;
    
    Mat() {
        std::fill(arr.begin(), arr.end(), static_cast<T>(0));
    }

    template <typename ...Ts>
    Mat(Ts... args) {
        static_assert(
            (sizeof...(Ts) == N * N && (true && ... && std::is_arithmetic_v<Ts>)) ||
            (sizeof...(Ts) == N &&
                ((true && ... && std::is_same_v<Ts, Vec<float, N>>) ||
                (true && ... && std::is_same_v<Ts, Vec<double, N>>))));
        
        if constexpr (sizeof...(Ts) == N * N) {
            arr = { static_cast<T>(args)... };
        }
        else {
            int i = 0;
            auto unwrap = [&](uint32_t i, auto v) {
                for (int j = 0; auto const& ve : v.arr)
                    arr[N * i + j++] = ve;
            };
            (unwrap(i++, args), ...);
        }
    }

    auto operator [](unsigned int idx) {
        return Column<T, N>(&arr[idx * N]);
    }

    auto operator [](unsigned int idx) const {
        return Column<T, N>(&arr[idx * N]);
    }

    auto transpose() const {
        Mat ret;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                ret[i][j] = arr[j * N + i];

        return ret;
    }

    auto operator *(const T scalar) const {
        Mat ret;
        for (int i = 0; i < N * N; i++)
            ret.arr[i] = arr[i] * scalar;

        return ret;
    }

    friend auto operator *(const T scalar, const Mat& mat) {
        return mat * scalar;
    }

    auto operator *=(const T scalar) {
        for (int i = 0; i < N * N; i++)
            arr[i] *= scalar;
    }

    auto operator *(const Vec<T, N>& vec) const {
        Vec<T, N> ret;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                ret[i] += arr[j * N + i] * vec[j];
        
        return ret;
    }

    auto operator *(const Vec<T, N - 1>& vec) const {
        Vec<T, N> tmp{vec, 0};
        Vec<T, N> ret;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                ret[i] += arr[j * N + i] * vec[j];
        
        return ret;
    }

    auto operator *(const Mat& mat) const {
        Mat ret;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++)
                    ret[i][j] += arr[k * N + j] * mat[i][k];

        return ret;
    }

    auto operator *=(const Mat& mat) {
        Mat tmp = *this * mat;
        *this = tmp;
        return *this;
    }

    operator OSL::Matrix44() const {
        // We only need this for Mat4
        static_assert(N == 4);

        OSL::Matrix44 ret;
        // Unfortunately imath matrix is row majored
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                ret[i][j] = arr[j * N + i];

        return ret;
    }

    auto inverse() const {
        if constexpr (N == 2) {
            T det = static_cast<T>(1) / (arr[0] * arr[3] - arr[1] * arr[2]);
            Mat ret{arr[3], -arr[1], -arr[2], arr[0]};
            return ret * det;
        }
        else if constexpr (N == 3) {
            T A = arr[4] * arr[8] - arr[7] * arr[5];
            T B = arr[7] * arr[2] - arr[1] * arr[8];
            T C = arr[1] * arr[5] - arr[4] * arr[2];
            T D = arr[6] * arr[5] - arr[3] * arr[8];
            T E = arr[0] * arr[8] - arr[6] * arr[2];
            T F = arr[3] * arr[2] - arr[0] * arr[5];
            T G = arr[3] * arr[7] - arr[6] * arr[4];
            T H = arr[6] * arr[4] - arr[0] * arr[7];
            T I = arr[0] * arr[4] - arr[3] * arr[1];

            T det = arr[0] * A + arr[3] * B + arr[6] * C;
            Mat ret{A, B, C, D, E, F, G, H, I};
            return ret * det;
        }
        else {
            // This segment is copied from pbrt-v3
            // Changed to column majored code
            int indxc[4], indxr[4];
            int ipiv[4] = {0, 0, 0, 0};
            Mat minv = *this;

            for (int i = 0; i < N; i++) {
                int irow = 0, icol = 0;
                T big = static_cast<T>(0);
                // Choose pivot
                for (int j = 0; j < N; j++) {
                    if (ipiv[j] != 1) {
                        for (int k = 0; k < N; k++) {
                            if (ipiv[k] == 0) {
                                if (std::abs(minv[k][j]) >= big) {
                                    big = static_cast<T>(std::abs(minv[k][j]));
                                    irow = j;
                                    icol = k;
                                }
                            }
                            else if (ipiv[k] > 1) {
                                std::cerr << "Singular matrix" << std::endl;
                            }
                        }
                    }
                }

                ++ipiv[icol];

                // Swap rows irow and icol for pivot
                if (irow != icol) {
                    for (int k = 0; k < N; k++)
                        std::swap(minv[k][irow], minv[k][icol]);
                }
                indxr[i] = irow;
                indxc[i] = icol;
                if (minv[icol][icol] == static_cast<T>(0))
                    std::cerr << "Singular matrix" << std::endl;

                // Set m[icol][icol] to one by scaling row icol
                float pivinv = static_cast<T>(1) / minv[icol][icol];
                minv[icol][icol] = static_cast<T>(1);
                for (int j = 0; j < N; j++) minv[j][icol] *= pivinv;

                // Subtract this row from others to zero out their columns
                for (int j = 0; j < N; j++) {
                    if (j != icol) {
                        T save = minv[icol][j];
                        minv[icol][j] = static_cast<T>(0);
                        for (int k = 0; k < N; k++) minv[k][j] -= minv[k][icol] * save;
                    }
                }
            }

            // Swap columns to reflect permutation
            for (int j = N - 1; j >= 0; j--) {
                if (indxr[j] != indxc[j]) {
                    for (int k = 0; k < N; k++)
                        std::swap(minv[indxr[j]][k], minv[indxc[j]][k]);
                }
            }

            return minv;
        }
    }

    friend std::ostream& operator <<(std::ostream& os, const Mat& mat) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                os << mat.arr[j * N + i] << " ";
            }
            os << std::endl;
        }

        return os;
    }

    static Mat identity() {
        Mat ret;
        for (int i = 0; i < N; i++)
            ret[i][i] = static_cast<T>(1);

        return ret;
    }

    static Mat scale(const Vec<T, N>& v) {
        Mat ret;
        for (int i = 0; i < N; i++)
            ret[i][i] = v[i];

        return ret;
    }

    static Mat translate(const Vec<T, N>& v) {
        static_assert(N > 2);

        Mat ret;
        for (int i = 0; i < N; i++) {
            ret[N - 1][i] = v[i];
            ret[i][i] = 1.f;
        }

        return ret;
    }

public:
    std::array<T, N * N> arr;
};

template <typename T>
using Mat2 = Mat<T, 2>;

template <typename T>
using Mat3 = Mat<T, 3>;

template <typename T>
using Mat4 = Mat<T, 4>;

using Mat2f = Mat2<float>;
using Mat2d = Mat2<double>;
using Mat3f = Mat3<float>;
using Mat3d = Mat3<double>;
using Mat4f = Mat4<float>;
using Mat4d = Mat4<double>;

template <typename T, uint N>
inline Mat<T, N> identity() {
    return Mat<T, N>::identity();
}

template <typename T, uint N>
inline Mat<T, N> transpose(const Mat<T, N>& m) {
    return m.transpose();
}

template <typename T, uint N>
inline Mat<T, N> inverse(const Mat<T, N>& m) {
    return m.inverse();
}

template <typename T, uint N>
inline Mat<T, N> translate(const Vec<T, N - 1>& v) {
    return Mat<T, N>::translate(Vec<T, N>{v, 0});
}

const auto translate3f = translate<float, 4>;

template <typename T, uint N>
inline Mat<T, N> rotate(const Vec<T, N - 1>& axis, const T& angle) {
    // FIXME : add rotate matrix calculation
    return Mat<T, N>::identity();
}

const auto rotate3f = rotate<float, 4>;

template <typename T, uint N>
inline Mat<T, N> scale(const Vec<T, N - 1>& v) {
    return Mat<T, N>::scale(Vec<T, N>{v, 1});
}

const auto scale3f = scale<float, 4>;

template <typename T>
OSL::Matrix44 to_osl_mat4(const Mat4<T>& m) {
    return static_cast<OSL::Matrix44>(m);
}

#endif

}