#pragma once

#include <cassert>

/*
 * Code in this file is copied from appleseed
 * https://github.com/appleseedhq/appleseed/blob/master/src/appleseed/foundation/math/mis.h
 */

template <typename T>
inline constexpr T mis_balance(const T q1, const T q2) {
    assert(q1 >= T(0.));
    assert(q2 >= T(0.));
    assert(q1 + q2 > T(0.));

    // Original implementation in as follows the same pattern.
    // Here we need to do 2 divs and 1 plus, not the most optimal
    // solution. Don't know if there's any other benefit out there.
    // The only reason I could imagine is about the floating point
    // calculation precision.

    // const T r2 = q2 / q1;
    // return T(1.) / (T(1.) + r2);

    return q1 / (q1 + q2);
}

template <typename T>
inline constexpr T mis_balance(const T q1, const T q2, const T q3) {
    assert(q1 >= T(0.));
    assert(q2 >= T(0.));
    assert(q3 >= T(0.));
    assert(q1 + q2 + q3 > T(0.));

    return q1 / (q1 + q2 + q3);
}

template <typename T>
inline constexpr T mis_power(const T q1, const T q2, const T q3, const T beta) {
    assert(q1 >= T(0.));
    assert(q2 >= T(0.));
    assert(q3 >= T(0.));
    assert(q1 + q2 + q3 > T(0.));

    assert(beta >= T(0.));

    // 3 divs, 2 pows, 2 plus
    /*
    const T r2 = q2 / q1;
    const T r3 = q3 / q1;
    const T r2_pow = std::pow(r2, beta);
    const T r3_pow = std::pow(r3, beta);

    return T(1.) / (T(1.) + r2_pow + r3_pow);
    */

    // Use this version for now
    // TODO : look into assembly code to verify which is better
    // 1 div, 3 pows, 2 plus
    const T r1 = std::pow(q1);
    const T r2 = std::pow(q2);
    const T r3 = std::pow(q3);

    return r1 / (r1 + r2 + r3);

    // If vectorized, hopefully...
    // 1 div, 1 pow, 1 plus(a.k.a the sum call)
    /*
    Vec3<T> v{q1, q2, q3};
    auto vpow = base::pow(v, beta);

    return vpow[0] / base::sum(vpow);
    */
}

template <typename T>
inline constexpr T mis_power2(const T q1, const T q2) {
    assert(q1 >= T(0.));
    assert(q2 >= T(0.));
    assert(q1 + q2 > T(0.));

    // 2 divs, 1 mult, 1 plus
    /*
    const T r2 = q2 / q1;
    const T r2_pow = r2 * r2;

    return T(1.) / (T(1.) + r2_pow);
    */

    // 1 div, 2 mult, 1 plus
    // Definitely a win... Why would appleseed use that code pattern...
    const T r1 = q1 * q1;
    const T r2 = q2 * q2;

    return r1 / (r1 + r2);
}

template <typename T>
inline constexpr T mis_power2(const T q1, const T q2, const T q3) {
    assert(q1 >= T(0.));
    assert(q2 >= T(0.));
    assert(q3 >= T(0.));
    assert(q1 + q2 + q3 > T(0.));

    // 3 divs, 2 mults, 2 plus
    /*
    const T r2 = q2 / q1;
    const T r3 = q3 / q1;
    const T r2_pow = r2 * r2;
    const T r3_pow = r3 * r3;

    return T(1.) / (T(1.) + r2_pow + r3_pow);
    */

    // 1 div, 3 mults, 2 plus
    const auto r1 = q1 * q1;
    const auto r2 = q2 * q2;
    const auto r3 = q3 * q3;

    return r1 / (r1 + r2 + r3);

    // Vectorized version
    // 1 div, 1 mult, 1 plus
    /*
    Vec3<T> v{q1, q2, q2};
    auto vpow = base::square(v);

    return vpow[0] / base::sum(vpow);
    */
}