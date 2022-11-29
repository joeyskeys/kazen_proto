#pragma once

#include <OSL/oslclosure.h>

#include "base/utils.h"
#include "core/accel.h"

class ShadingContext {
    void*   data;
    OSL::ShaderGlobals* sg,
    void*   closure_sample;
    Accel*  accel;
    Frame   frame;
};