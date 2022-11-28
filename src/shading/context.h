#pragma once

#include <OSL/oslclosure.h>

#include "base/accel.h"

class ShadingContext {
    void*   data;
    OSL::ShaderGlobals* sg,
    void*   closure_sample;
    Accel*  accel;
};