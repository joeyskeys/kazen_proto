#pragma once

#include <OSL/oslexec.h>
#include <OSL/rendererservices.h>
#include <OpenImageIO/texture.h>

#include "core/hitable.h"

void register_closures(OSL::ShadingSystem *shadingsys);

class KazenRenderServices : public OSL::RendererServices
{
public:
    using Transformation = OSL::Matrix44;

    KazenRenderServices(OIIO::TextureSystem *texture_sys, HitablePtr p);
    ~KazenRenderServices() {}

private:
    int supports(OIIO::string_view feature) const override;

    bool get_matrix(
        OSL::ShaderGlobals *sg,
        OSL::Matrix44 &result,
        OSL::TransformationPtr xform,
        float time) override;

    bool get_inverse_matrix(
        OSL::ShaderGlobals *sg,
        OSL::Matrix44 &result,
        OSL::TransformationPtr xform,
        float time) override;

    bool get_matrix(
        OSL::ShaderGlobals *sg,
        OSL::Matrix44 &result,
        OSL::TransformationPtr xform) override;

    bool get_inverse_matrix(
        OSL::ShaderGlobals *sg,
        OSL::Matrix44 &result,
        OSL::TransformationPtr xform) override;

    /*
    // Figure out how to use named transform later
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

    void name_transform(const char *name, const Transformation & xform);

    bool transform_points(
        OSL::ShaderGlobals *sg,
        OIIO::ustring from,
        OIIO::ustring to,
        float time,
        const OSL::Vec3 *Pin,
        OSL::Vec3 *Pout,
        int npoints,
        OSL::TypeDesc::VECSEMANTICS vectype) override;
    */

    bool trace(
        TraceOpt &options,
        OSL::ShaderGlobals *sg,
        const OSL::Vec3 &P,
        const OSL::Vec3 &dPdx,
        const OSL::Vec3 &dPdy,
        const OSL::Vec3 &R,
        const OSL::Vec3 &dRdx,
        const OSL::Vec3 &dRdy) override;

    bool getmessage(
        OSL::ShaderGlobals *sg,
        OIIO::ustring source,
        OIIO::ustring name,
        OIIO::TypeDesc type,
        void *val,
        bool derivatives) override;

    bool get_attribute(
        OSL::ShaderGlobals *sg,
        bool derivatives,
        OIIO::ustring object,
        OIIO::TypeDesc type,
        OIIO::ustring name,
        void *val) override;

    bool get_array_attribute(
        OSL::ShaderGlobals *sg,
        bool derivatives,
        OIIO::ustring object,
        OIIO::TypeDesc type,
        OIIO::ustring name,
        int index,
        void *val) override;

    bool get_userdata(
        bool derivatives,
        OIIO::ustring name,
        OIIO::TypeDesc type,
        OSL::ShaderGlobals *sg,
        void *val) override;

    static void globals_from_hit(
        OSL::ShaderGlobals& sg,
        const Ray& r,
        const float& t,
        bool flip=false);

public:
    std::vector<OSL::ShaderGroupRef> shaders;

private:
    OIIO::TextureSystem *texture_sys;
    HitablePtr accel_ptr;
};