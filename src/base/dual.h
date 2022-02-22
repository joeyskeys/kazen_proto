#pragma once

#define USE_OSL_DUAL

#ifndef USE_OSL_DUAL

#else
#include <OSL/dual.h>
#include <OSL/dual_vec.h>

#include "base/vec.h"

using OSL::Dual;
using OSL::Dual2;

template <typename T>
using Dual3 = Dual<T, 3>;

#endif

using Dual2f    = Dual2<float>;
using Dual3f    = Dual3<float>;
using Dual2V3f  = Dual2<Vec3f>;
using Dual2V3d  = Dual2<Vec3d>;
using Dual3V3f  = Dual3<Vec3f>;
using Dual3V3d  = Dual3<Vec3d>;

template <typename T, int P>
inline Dual<Vec3<T>, P> normalize(const Dual<Vec3<T>, P>& a) {
    auto ax = a.x;
    auto ay = a.y;
    auto az = a.z;
    auto len = sqrt(ax * ax + ay * ay + az * az);
    if (len > Vec3<T>::ValueType{0}) {
        auto invlen = Vec3<T>::ValueType{1} / len;
        auto nax = ax * invlen;
        auto nay = ay * invlen;
        auto naz = az * invlen;
        return Vec3<T>(nax, nay, naz);
    }
    else
        return Vec3<T>(0, 0, 0);
}