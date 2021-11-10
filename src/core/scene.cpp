#include <cstring>
#include <sstream>
#include <stdexcept>

#include <frozen/set.h>
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

    template <int N>
    using AttrSet = forzen::set<frozen::string, N>;

    constexpr AttrSet<5> camera_attributes = {
        "resolution",
        "position",
        "lookat",
        "up",
        "fov"
    };

    template <typename T>
    bool parse_components(const std::string& attrstr, const std::string& typestr std::stringstream& ss, void *dst) {
        auto found = attrstr.find(typestr);
        if (found != std::string::npos) {
            int type_string_size = typestr.size();
            int cnt = 1;
            if (attrstr.size() > type_string_size)
                cnt = attrstr[type_string_size] - '0';
            std::array<T, cnt> buf;
            for (int i = 0; i < cnt; i++)
                ss >> buf[i];
            memcpy(dst, &buf, cnt * sizeof(T));
            return true;
        }
        return false;
    }

    void parse_attribute(pugixml::xml_attribute& attr, auto& attr_set) {
        auto it = attr_set.find(attr.name());
        std::string typestr;
        if (it != attr_set.end()) {
            ss.str(attr.value());
            ss >> typestr;
            int cnt = 1;

            // float  type
            if (parse_components<float>(typestr, "float", ss, camera->address_of(attr.name())))
                return;
            // int type
            if (parse_components<int>(typestr, "int", ss, camera->address_of(attr.name())))
                return;
        }
    }

    auto root_tag = gettag(node);
    if (root_tag == EScene) {
        // Found th root scene node, parse elements
        for (auto& ch : node.children()) {
            auto child_tag = gettag(ch);
            std::stringstream ss;
            // Now we assume all attribute component is float type..
            std::string type_str;
            switch (child_tag) {
                case ECamera:
                    for (auto& attr : node.attributes())
                        parse_attribute(attr, camera_attributes);
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