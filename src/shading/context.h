#pragma once

#include <OSL/oslclosure.h>

#include "base/utils.h"
#include "core/accel.h"
#include "core/intersection.h"

// Need to refactor the structure and make clear that where does each piece of information
// store.
// Currently there is a overlap in isect_i and sg, but since both are pointer here, doesn't
// really affect much.
// Frame is not needed, isect_i contains the frame of incoming point, closure_sample for
// BSSRDF contains the frame of outcoming point.
struct ShadingContext {
    void*               data;
    OSL::ShaderGlobals* sg;
    void*               closure_sample;
    Accelerator*        accel;
    Intersection*       isect_i;
    std::unordered_map<std::string, OSL::ShaderGroupRef>* shaders;
};