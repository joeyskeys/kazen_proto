#pragma once

#include "base/vec.h"

class RGBSpectrum : public Vec3f {
public:
    inline float& r() {
        return arr[0];
    }

    inline float& g() {
        return arr[1];
    }

    inline float& b() {
        return arr[2];
    }
};
