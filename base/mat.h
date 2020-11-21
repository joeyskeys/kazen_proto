#pragma once

#include <array>

#include "vec.h"

template <typename T, unsigned int N>
class column {
public:
    template <typename T>
    column(T* d)
        : data(d)
    {}

    auto operator [](unsigned int idx) {
        // No bound checking for now
        return data[idx];
    }
};

private:
    T* data;
};

template <typename T, unsigned int N>
class mat {
public:
    template <typename ...Ts>
    mat(Ts... args) {
        static_assert(sizeof...(Ts) == N * N);
        v = { static_cast<T>(args)... };
    }

    auto operator [](unsigned int idx) {
        return column<T>(&arr[idx * N]);
    }

private:
    std::array<T, N * N> arr;
};
