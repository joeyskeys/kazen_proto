#include "renderservices.h"

#include "core/intersection.h"
#include "core/ray.h"
#include "core/transform.h"

KazenRenderServices::KazenRenderServices(OIIO::TextureSystem *tex_sys, AccelPtr p)
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
    result = transform.mat;

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
    result = transform.mat_inv;

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
    result = transform.mat;

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
    result = transform.mat_inv;

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
    const Ray r(p, d);

    // TODO : find out if trace data has to be OSL relavent, aka using
    // OpenEXR types
    auto traced_isect_ptr =
        reinterpret_cast<Intersection*>(sg->tracedata);
    Intersection isect;
    if (accel_ptr->intersect(r, &isect)) {
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