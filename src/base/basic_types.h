#pragma once

#include <limits>

using uint = unsigned int;

template <typename T>
constexpr T epsilon = static_cast<T>(0.000001);

constexpr size_t INVALID_ID = std::numeric_limits<size_t>::max();