
#include "base/utils.h"
#include "base/vec.h"
#include "shading/bsdf.h"
#include "shading/bssrdfs.h"
#include "shading/context.h"

static bool find_po(ShadingContext* ctx, bssrdf_profile_sample_func& profile_func, const Vec4f& rand) {
    auto dipole_params = reinterpret_cast<KpDipoleParams*>(ctx->data);
    auto bssrdf_sample = reinterpret_cast<BSSRDFSample*>(ctx->closure_sample);
    const float disk_radius = profile_func(ctx.data, rand[0]);

    if (disk_radius == 0.f)
        return false;

    if (disk_radius >= dipole_params->max_radius)
        return false;

    auto phi = constants::two_pi<float>() * rand[1];
    auto disk_point = Vec3f{disk_radius * std::cos(phi), 0.f, disk_radius * std::sin(phi)};

    // Choose a projection axis
    Frame frame;
    if (rand[2] < 0.5f) {
        frame = Frame(ctx->frame.s, ctx->frame.t, ctx->frame.n);
    }
    else if (rand[2] < 0.75f) {
        frame = Frame(ctx->frame.t, ctx->frame.n, ctx->frame.s);
    }
    else {
        frame = Frame(ctx->frame.n, ctx->frame.s, ctx->frame.t);
    }

    auto h = std::sqrt(square(dipole_params->max_radius) - square(disk_radius));
    auto hn = h * frame.n;
    auto entry_pt = disk_point + hn;
    auto ray_dir = base::normalize(-hn);

    const static int max_intersection_cnt = 10;
    int found_intersection = 0;
    Intersection isects[max_intersection_cnt];
    auto start_pt = entry_pt;
    for (int i = 0; i < max_intersection_cnt; i++) {
        Ray r(start_pt, ray_dir);
        ctx->accel->intersect(r, isects[i]);
        start_pt = isects[i].P;
        ++found_intersection;
    }

    // Randomly chose one intersection
    if (found_intersection == 0)
        return false;
    else if (found_intersection == 1) {
        bssrdf_sample->po = isects[0].P;
        return true;
    }
    else {
        uint idx = rand[3] * max_intersection_cnt * 0.999999f;
        bssrdf_sample->po = isects[idx];
        return true;
    }
}

static bool sample_dipole(ShadingContext* ctx, bssrdf_profile_sample_func& profile_func, const Vec4f& rand) {
    auto found_po = find_po(ctx, profile_func, rand);
    if (!found_po)
        return false;

    auto bssrdf_sample = reinterpret_cast<BSSRDFSample*>(ctx->closure_sample);
    // We need to unify the interfaces now
    auto f = bssrdf_sample->sampled_brdf->sample(ctx, base::head<3>(rand));
    return f;
}

RGBSpectrum KpDipole::eval(ShadingContext* ctx) {
    
}

RGBSpectrum KpDipole::sample(ShadingContext* ctx, const Vec4f& rand) {

}