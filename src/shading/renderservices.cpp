#include "renderservices.h"

#include "core/transform.h"

KazenRenderServices::KazenRenderServices(OIIO::TextureSystem *tex_sys)
    : texture_sys(tex_sys)
{
}

bool KazenRenderServices::get_matrix(
    OSL::ShaderGlobals *sg,
    OSL::Matrix44 &result,
    OSL::TransformationPtr xform,
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
    OSL::TransformationPtr xform,
    float time)
{
    if (!xform)
        return false;

    auto transform = reinterpret_cast<const Transform*>(xform);
    result = transform.mat_inv;
}

bool get_matrix(
    OSL::ShaderGlobals *sg,
    OSL::Matrix44 &result,
    OSL::TransformationPtr xform) override;

bool get_inverse_matrix(
    OSL::ShaderGlobals *sg,
    OSL::Matrix44 &result,
    OSL::TransformationPtr xform) override;

bool get_matrix(
    OSL::ShaderGlobals *sg,
    OSL::Matrix44 &result,
    OIIO::ustring from,
    float time) override;

bool get_inverse_matrix(
    OSL::ShaderGlobals *sg,
    OSL::Matrix44 &result,
    OIIO::ustring to,
    float time) override;

bool get_matrix(
    OSL::ShaderGlobals *sg,
    OSL::Matrix44 &result,
    OIIO::ustring from) override;

bool get_inverse_matrix(
    OSL::ShaderGlobals *sg,
    OSL::Matrix44 &result,
    OIIO::ustring to) override;
