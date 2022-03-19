#pragma once

#include <cstddef>
#include <limits>

using uint = unsigned int;

template <typename T>
constexpr T epsilon = static_cast<T>(1e-4f);

constexpr size_t INVALID_ID = std::numeric_limits<size_t>::max();