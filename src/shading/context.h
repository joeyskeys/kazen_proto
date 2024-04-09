#pragma once

#include <unordered_map>

#include <OSL/oslclosure.h>
#include <OSL/oslexec.h>

#include "base/utils.h"
#include "core/accel.h"
#include "core/intersection.h"

using ShaderMap = std::unordered_map<std::string, OSL::ShaderGroupRef>;
using PTXMap = std::unordered_map<std::string, std::string>;

struct ShadingEngine {
    OSL::ShadingSystem*     osl_shading_sys;
    OSL::PerThreadInfo*     osl_thread_info;
    OSL::ShadingContext*    osl_shading_ctx;
    OSL::ShaderGroupRef     background_shader;
    ShaderMap*              shaders;

    inline void execute(const std::string& shader_name, OSL::ShaderGlobals& sg) const {
        auto shader_ptr = shaders->at(shader_name);
        if (!shader_ptr)
            throw std::runtime_error(fmt::format("Shader for name : {} does not exists",
                shader_name));
        osl_shading_sys->execute(*osl_shading_ctx, *shader_ptr, sg);
    }

    inline void execute(OSL::ShaderGroupRef shader, OSL::ShaderGlobals& sg) const {
        osl_shading_sys->execute(*osl_shading_ctx, *shader, sg);
    }

    PTXMap gen_ptx() {
        PTXMap ptx_map;
        std::vector<const char*> outputs { "Cout" };
        std::vector<void*> material_interactive_params;
        for (const auto&[name, groupref] : *shaders) {
            std::string group_name, fused_name;
            osl_shading_sys->getattribute(groupref.get(), "groupname", group_name);
            osl_shading_sys->getattribute(groupref.get(), "group_fused_name",
                fused_name);
            osl_shading_sys->attribute(groupref.get(), "renderer_outputs",
                OSL::TypeDesc(OSL::TypeDesc::STRING, outputs.size()), outputs.data());
            osl_shading_sys->optimize_group(groupref.get(), nullptr);

            if (!osl_shading_sys->find_symbol(*groupref.get(), OSL::ustring(outputs[0]))) {
                throw std::runtime_error(fmt::format("requested output '{}', which wasn't found", outputs[0]));
            }

            // Retrieve the compiled ShaderGroup PTX
            std::string osl_ptx;
            osl_shading_sys->getattribute(groupref.get(), "ptx_compiled_version",
                OSL::TypeDesc::PTR, &osl_ptx);
            if (osl_ptx.empty()) {
                throw std::runtime_error(fmt::format("failed to generate PTX for shadergroup {}",
                    group_name));
            }

            // TODO: Save PTX if necessary

            // TODO: Save the params in another place
            void* interactive_params = nullptr;
            osl_shading_sys->getattribute(groupref.get(), "device_interactive_params",
                OSL::TypeDesc::PTR, &interactive_params);
            
            ptx_map.emplace(name, std::move(osl_ptx));
        }

        return ptx_map;
    }
};

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
    Ray*                ray;
    Intersection*       isect_i;
    Intersection        isect_o;
    //std::unordered_map<std::string, OSL::ShaderGroupRef>* shaders;
    ShadingEngine*      engine_ptr;
};