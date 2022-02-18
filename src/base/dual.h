#pragma once

#define USE_OSL_DUAL

#ifndef USE_OSL_DUAL

#else
#include <OSL/dual.h>
#include <OSL/dual_vec.h>

using OSL::Dual2;

template<class T>
using Dual3 = Dual<T, 3>;

#endif