#pragma once

#include "base/vec.h"

class RGBSpectrum : public Vec3f {
public:
    RGBSpectrum()
    {
        std::fill(arr.begin(), arr.end(), 0.f);
    }

    RGBSpectrum(float r, float g, float b)
    {
        arr[0] = r;
        arr[1] = g;
        arr[2] = b;
    }

    inline float& r()
    {
        return arr[0];
    }

    inline const float r() const
    {
        return arr[0];
    }

    inline float& g()
    {
        return arr[1];
    }

    inline const float g() const
    {
        return arr[1];
    }

    inline float& b()
    {
        return arr[2];
    }

    inline const float b() const
    {
        return arr[2];
    }
};
