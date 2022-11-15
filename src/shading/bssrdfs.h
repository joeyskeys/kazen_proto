#pragma once

#include <OSL/oslclosure.h>

#include "shading/bssrdfs.h"

using OSL::TypeDesc;

/* BSSRDF is new subject, starting with pbrt and appleseed references:
 * https://www.pbr-book.org/3ed-2018/Volume_Scattering/The_BSSRDF
 * http://library.imageworks.com/pdfs/imageworks-library-BSSRDF-sampling.pdf
 * https://www.solidangle.com/research/s2013_bssrdf_slides.pdf
 */

// Diffusion approximation of subsurface reflection
// Checkout "A Practical Model for Subsurface Light Transport" page 3 formular 5
// https://graphics.stanford.edu/papers/bssrdf/bssrdf.pdf
static RGBSpectrum Sd(const Vec3f& pi, const Vec3f& wi, const Vec3f& po,
    const Vec3f& wo)
{

}

struct DipoleParams {

};