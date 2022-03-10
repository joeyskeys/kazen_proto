#pragma once

#include <cstdint>
#include <cstddef>

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