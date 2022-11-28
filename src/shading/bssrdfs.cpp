
#include "shading/bssrdfs.h"
#include "shading/context.h"

static bool find_po(ShadingContext* ctx, bssrdf_profile_sample_func& profle_func, const Vec3f& rand) {
    const float disk_radius = profle_func(ctx.data, rand[0]);
}

RGBSpectrum KpDipole::eval(ShadingContext* ctx) {
    
}

RGBSpectrum KpDipole::sample(ShadingContext* ctx, const Vec3f& rand) {

}