#pragma once

#include "shading/bsdfs.h"

float weak_white_furnace_test(const bsdf_eval_func& func,
    const uint32_t phi_span, const uint32_t theta_span,
    const float alpha, const float theta_o);