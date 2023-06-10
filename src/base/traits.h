#pragma once

#include <cstdint>
#include <cstddef>

#include "base/vec.h"

// Code copied from appleseed/foundation/utility/typetraits.h with original
// MIT license, authored by Francois Beaune, Jupiter Jazz
// Conversion of a numeric type to another numeric type of the same size.

template <size_t n>
struct TypeConvImpl;

template <>
struct TypeConvImpl<4> {
    using Int = std::int32_t;
    using UInt = std::uint32_t;
    using Scalar = float;
};

template <>
struct TypeConvImpl<8> {
    using Int = std::int64_t;
    using UInt = std::uint64_t;
    using Scalar = double;
};

template <typename T>
struct TypeConv : public TypeConvImpl<sizeof(T)> {};

template <typename T>
struct is_bool {
    static constexpr bool value = false;
};

template <>
struct is_bool<bool> {
    static constexpr bool value = true;
};

template <typename T>
inline constexpr bool is_bool_v = is_bool<T>::value;

template <typename T>
struct is_float {
    static constexpr bool value = false;
};

template <>
struct is_float<float> {
    static constexpr bool value = true;
};

template <typename T>
inline constexpr bool is_float_v = is_float<T>::value;

template <typename T>
struct is_int {
    static constexpr bool value = false;
};

template <>
struct is_int<int> {
    static constexpr bool value = true;
};

template <typename T>
inline constexpr bool is_int_v = is_int<T>::value;

template <typename T>
struct is_vec3f {
    static constexpr bool value = false;
};

template <>
struct is_vec3f<base::Vec3f> {
    static constexpr bool value = true;
};

template <typename T>
inline constexpr bool is_vec3f_v = is_vec3f<T>::value;

template <typename T>
struct is_vec4f {
    static constexpr bool value = false;
};

template <>
struct is_vec4f<base::Vec4f> {
    static constexpr bool value = true;
};

template <typename T>
inline constexpr bool is_vec4f_v = is_vec4f<T>::value;

template <typename T>
struct is_str {
    static constexpr bool value = false;
};

template <>
struct is_str<std::string> {
    static constexpr bool value = true;
};

template <typename T>
inline constexpr bool is_str_v = is_str<T>::value;

template <typename T>
struct is_ustr {
    static constexpr bool value = false;
};

template <>
struct is_ustr<OSL::ustring> {
    static constexpr bool value = true;
};

template <typename T>
inline constexpr bool is_ustr_v = is_ustr<T>::value;
