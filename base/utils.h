#pragma once

#include <cmath>

template <typename T>
T to_radian(const T degree) {
    return static_cast<T>(degree / 180. * M_PI);
}

template <typename T>
T to_degree(const T radian) {
    return static_cast<T>(radian / M_PI * 180.);
}