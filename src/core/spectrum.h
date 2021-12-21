#pragma once

#include "base/vec.h"

class RGBSpectrum : public Vec3f {
public:
    RGBSpectrum() {
        std::fill(arr.begin(), arr.end(), 0.f);
    }

    RGBSpectrum(float v) {
        std::fill(arr.begin(), arr.end(), v);
    }

    RGBSpectrum(float r, float g, float b) {
        arr[0] = r;
        arr[1] = g;
        arr[2] = b;
    }

    RGBSpectrum(Vec3f vec) {
        arr[0] = vec.x();
        arr[1] = vec.y();
        arr[2] = vec.z();
    }

    inline float& r() {
        return arr[0];
    }

    inline const float r() const {
        return arr[0];
    }

    inline float& g() {
        return arr[1];
    }

    inline const float g() const {
        return arr[1];
    }

    inline float& b() {
        return arr[2];
    }

    inline const float b() const {
        return arr[2];
    }

    /*
    RGBSpectrum operator *(const float scalar) const {
        return RGBSpectrum(arr[0] * scalar, arr[1] * scalar, arr[2] * scalar);
    }

    friend RGBSpectrum operator *(const float scalar, const RGBSpectrum& spec) {
        return spec * scalar;
    }
    */
};
