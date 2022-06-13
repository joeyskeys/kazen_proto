#include <cstring>

#include "renderservices.h"
#include "core/intersection.h"
#include "core/ray.h"
#include "core/shape.h"
#include "core/transform.h"

KazenRenderServices::KazenRenderServices()
    : texture_sys(nullptr)
    , accel_ptr(nullptr)
{

}

KazenRenderServices::KazenRenderServices(OIIO::TextureSystem *tex_sys, HitablePtr p)
    : texture_sys(tex_sys)
    , accel_ptr(p)
{
}

bool KazenRenderServices::get_matrix(
    OSL::ShaderGlobals *sg,
    OSL::Matrix44 &result,
    OSL::TransformationPtr xform,
    float time)
{
    // Ignore time parameter for now since we don't have motion blur yet
    // Same for similar functions
    if (!xform)
        return false;

    auto transform = reinterpret_cast<const Transform*>(xform);
    result = transform->mat;

    return true;
}

bool KazenRenderServices::get_inverse_matrix(
    OSL::ShaderGlobals *sg,
    OSL::Matrix44 &result,
    OSL::TransformationPtr xform,
    float time)
{
    if (!xform)
        return false;

    auto transform = reinterpret_cast<const Transform*>(xform);
    result = transform->mat_inv;

    return true;
}

bool KazenRenderServices::get_matrix(
    OSL::ShaderGlobals *sg,
    OSL::Matrix44 &result,
    OSL::TransformationPtr xform)
{
    if (!xform)
        return false;

    auto transform = reinterpret_cast<const Transform*>(xform);
    result = transform->mat;

    return true;
}

bool KazenRenderServices::get_inverse_matrix(
    OSL::ShaderGlobals *sg,
    OSL::Matrix44 &result,
    OSL::TransformationPtr xform)
{
    if (!xform)
        return false;

    auto transform = reinterpret_cast<const Transform*>(xform);
    result = transform->mat_inv;

    return true;
}

/*
bool KazenRenderServices::get_matrix(
    OSL::ShaderGlobals *sg,
    OSL::Matrix44 &result,
    OIIO::ustring from,
    float time)
{
    if (!xform)
        return false;

    auto transform = reinterpret_cast<const Transform*>(xform);
    result = transform.mat;

    return true;
}

bool KazenRenderServices::get_inverse_matrix(
    OSL::ShaderGlobals *sg,
    OSL::Matrix44 &result,
    OIIO::ustring to,
    float time)
{
    if (!xform)
        return false;

    auto transform = reinterpret_cast<const Transform*>(xform);
    result = transform.mat_inv;

    return true;
}

bool KazenRenderServices::get_matrix(
    OSL::ShaderGlobals *sg,
    OSL::Matrix44 &result,
    OIIO::ustring from)
{
    if (!xform)
        return false;

    auto transform = reinterpret_cast<const Transform*>(xform);
    result = transform.mat;

    return true;
}

bool KazenRenderServices::get_inverse_matrix(
    OSL::ShaderGlobals *sg,
    OSL::Matrix44 &result,
    OIIO::ustring to)
{
    if (!xform)
        return false;

    auto transform = reinterpret_cast<const Transform*>(xform);
    result = transform.mat_inv;

    return true;
}

bool KazenRenderServices::transform_points(
    OSL::ShaderGlobals *sg,
    OIIO::ustring from,
    OIIO::ustring to,
    float time,
    const OSL::Vec3 *Pin,
    OSL::Vec3 *Pout,
    int npoints,
    OSL::TypeDesc::VECSEMANTICS vectype)
{

}

*/

bool KazenRenderServices::trace(
    TraceOpt &options,
    OSL::ShaderGlobals *sg,
    const OSL::Vec3 &P,
    const OSL::Vec3 &dPdx,
    const OSL::Vec3 &dPdy,
    const OSL::Vec3 &R,
    const OSL::Vec3 &dRdx,
    const OSL::Vec3 &dRdy)
{
    // Get intersection info from raw pointer
    auto isect_ptr = reinterpret_cast<Intersection*>(sg->tracedata);

    const Vec3f o(P);
    const Vec3f d(R);
    const Ray r(P, d);

    // TODO : find out if trace data has to be OSL relavent, aka using
    // OpenEXR types
    auto traced_isect_ptr =
        reinterpret_cast<Intersection*>(sg->tracedata);
    Intersection isect;
    if (accel_ptr->intersect(r, isect)) {
        *traced_isect_ptr = isect;
        return true;
    }

    return false;
}

bool KazenRenderServices::getmessage(
    OSL::ShaderGlobals *sg,
    OIIO::ustring source,
    OIIO::ustring name,
    OIIO::TypeDesc type,
    void *val,
    bool derivatives)
{
    return false;
}

bool KazenRenderServices::get_attribute(
    OSL::ShaderGlobals *sg,
    bool derivatives,
    OIIO::ustring object,
    OIIO::TypeDesc type,
    OIIO::ustring name,
    void *val)
{
    return false;
}

bool KazenRenderServices::get_array_attribute(
    OSL::ShaderGlobals *sg,
    bool derivatives,
    OIIO::ustring object,
    OIIO::TypeDesc type,
    OIIO::ustring name,
    int index,
    void *val)
{
    return false;
}

bool KazenRenderServices::get_userdata(
    bool derivatives,
    OIIO::ustring name,
    OIIO::TypeDesc type,
    OSL::ShaderGlobals *sg,
    void *val)
{
    return false;
}

void KazenRenderServices::globals_from_hit(
    OSL::ShaderGlobals& sg,
    const Ray& r,
    const Intersection& isect,
    bool flip)
{
    memset((char*)&sg, 0, sizeof(OSL::ShaderGlobals));
    sg.P = isect.P;
    /*
    sg.N = isect.shading_normal;
    sg.Ng = isect.N;
    */
    sg.N = OSL::Vec3(0.f, 1.f, 0.f);
    sg.Ng = isect.to_local(isect.N);

    sg.u = isect.uv[0];
    sg.v = isect.uv[1];

    sg.surfacearea = isect.shape->area();

    // TODO : setup dPdu dPdv correctly
    sg.dPdu = isect.tangent;
    sg.dPdv = isect.bitangent;

    sg.I = isect.to_local(r.direction);
    sg.backfacing = sg.N.dot(sg.I) > 0;
    if (sg.backfacing) {
        //sg.N = -sg.N;
        //sg.Ng = -sg.Ng;
    }
    sg.flipHandedness = flip;
    sg.renderstate = &sg;
}

void KazenRenderServices::globals_from_lightrec(
    OSL::ShaderGlobals& sg,
    const LightRecord& lrec)
{
    memset((char*)&sg, 0, sizeof(OSL::ShaderGlobals));
    sg.P = lrec.lighting_pt;
    sg.N = sg.Ng = OSL::Vec3(0.f, 1.f, 0.f);

    sg.u = lrec.uv[0];
    sg.v = lrec.uv[1];

    // Ignore partial derivatives for now

    sg.surfacearea = lrec.area;
    sg.flipHandedness = false;
    sg.renderstate = &sg;
}