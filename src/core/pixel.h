#pragma once

#include <limits>
#include <array>
#include <algorithm>

#include "spectrum.h"

class Pixel {
public:
    template<typename T>
    auto convert_to() {
        auto type_convert = [](float c) {
            return std::clamp(
                std::numeric_limits<T>::max() * c,
                T{0},
                std::numeric_limits<T>::max() - 1);
        };

        return std::array<T, 3>{
            type_convert(r),
            type_convert(g),
            type_convert(b)
        };
    }

    Pixel& operator=(const RGBSpectrum& s) {
        r = s[0];
        g = s[1];
        b = s[2];

        return *this;
    }

    float r = 0.f;
    float g = 0.f;
    float b = 0.f;
};