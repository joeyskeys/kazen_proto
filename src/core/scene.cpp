#include <stdexcept>

#include <frozen/unordered_map.h>
#include <pugixml.hpp>

#include "scene.h"

void Scene::parse_from_file(fs::path file_path) {
    // Some code copied from nori:
    // https://github.com/wjakob/nori.git

    pugi::xml_document doc;
    pugi::xml_parse_result ret = doc.load_file(file_path.c_str());

        /* Helper function: map a position offset in bytes to a more readable line/column value */
    auto offset = [&](ptrdiff_t pos) -> std::string {
        std::fstream is(filename);
        char buffer[1024];
        int line = 0, linestart = 0, offset = 0;
        while (is.good()) {
            is.read(buffer, sizeof(buffer));
            for (int i = 0; i < is.gcount(); ++i) {
                if (buffer[i] == '\n') {
                    if (offset + i >= pos)
                        return tfm::format("line %i, col %i", line + 1, pos - linestart);
                    ++line;
                    linestart = offset + i;
                }
            }
            offset += (int) is.gcount();
        }
        return "byte offset " + std::to_string(pos);
    };

    if (!result) /* There was a parser / file IO error */
        throw std::runtime_error("Error while parsing \"%s\": %s (at %s)", filename, result.description(), offset(result.offset));

    enum ETag {
        EScene,
        ECamera,
        EAccelerator,
        // objects
        EObjects,
        ESphere,
        ETriangle,
        // materials
        EMaterials,
        EShaderGroup,
        EShader,
        EConnectShaders,
        // lights
        ELights,
        EPointLight,
        EInvalid
    };

    constexpr frozen::unordered_map<frozen::string, ETag, 2> tags = {
        {"Scene", EScene},
        {"Camera", ECamera},
        {"Accelerator", EAccelerator},
        {"Objects", EObjects},
        {"Sphere", ESphere},
        {"Triangle", ETriangle},
        {"Materials", EMaterials},
        {"ShaderGroup", EShaderGroup},
        {"Shader", EShader},
        {"ConnectShaders", EConnectShaders},
        {"Lights", ELights},
        {"PointLight", EPointLight}
    };

    auto& node = *doc.begin();
    // Skip over comments
    while (node.type() == pugi::node_comment || node.type() == pugi::node_declaration)
        ++node;

    if (node.type() != pugi::node_element)
        throw std::runtime_error(
            "Error while parsing \"%s\": unexpected content at %s", file_path, offset(node.offset_debug()));

    auto gettag = [&file_path](pugi::xml_node &node) {
        auto it = tags.find(node.name());
        if (it == tags.end())
            throw std::runtime_error("Error while parsing \"%s\": unexpected tag \"%s\" at %s",
                file_path, node.name(), offset(node.offset_debug()));
        int tag = it->second;
        return tag;
    }

    constexpr frozen::unordered_map<frozen::string, frozen::string, 2> camera_attributes = {
        {"resolution", "float2 %f %f"},
        {"position", "float3 %f %f %f"},
        {"lookat", "float3 %f %f %f"},
        {"up", "float3 %f %f %f"},
        {"fov", "float %f"}
    };

    auto root_tag = gettag(node);
    if (root_tag == EScene) {
        // Found th root scene node, parse elements
        for (auto& ch : node.children()) {
            auto child_tag = gettag(ch);
            switch (child_tag) {
                case ECamera:
                    for (auto& attr : node.attributes()) {
                        
                    }
                    break;

                case EAccelerator:
                    break;

                case EObjects:
                    break;

                case EMaterials:
                    break;

                case ELights:
                    break;

                default:
                    break;
            }
        }
    }
}