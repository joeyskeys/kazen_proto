#include <cstring>
#include <iostream>
#include <fstream>
#include <stdexcept>

#include <fmt/core.h>
#include <frozen/set.h>
#include <frozen/string.h>
#include <frozen/unordered_map.h>
#include <pugixml.hpp>

#include "scene.h"

template <int N>
using AttrSet = frozen::set<frozen::string, N>;

template <typename T>
bool parse_components(const std::string& attrstr, std::stringstream& ss, void* dst) {
    auto found = attrstr.find(TypeInfo<float>::name);
    if (found != std::string::npos) {
        int namelength = TypeInfo<float>::namelength;
        int cnt = 1;
        if (attrstr.size() > namelength)
            cnt = attrstr[namelength] - '0';

        // Assume we have 4 components at most
        std::array<T, 4> buf;
        for (int i = 0; i < cnt; i++)
            ss >> buf[i];
        memcpy(dst, &buf, cnt * sizeof(T));
        return true;
    }
    return false;
}

template <typename T>
void parse_attribute(std::stringstream& ss, pugi::xml_attribute attr, DictLike* obj, T& attr_set) {
    auto it = attr_set.find(frozen::string(attr.name()));
    std::string typestr;
    if (it != attr_set.end()) {
        ss.str(attr.value());
        ss >> typestr;
        int cnt = 1;

        // float  type
        if (parse_components<float>(typestr, ss, obj->address_of(attr.name())))
            return;
        // int type
        if (parse_components<int>(typestr, ss, obj->address_of(attr.name())))
            return;
    }
}

void Scene::parse_from_file(fs::path filepath) {
    // Some code copied from nori:
    // https://github.com/wjakob/nori.git

    pugi::xml_document doc;
    pugi::xml_parse_result ret = doc.load_file(filepath.c_str());

    /* Helper function: map a position offset in bytes to a more readable line/column value */
    auto offset = [&filepath](ptrdiff_t pos) -> std::string {
        std::fstream is(filepath.c_str());
        char buffer[1024];
        int line = 0, linestart = 0, offset = 0;
        while (is.good()) {
            is.read(buffer, sizeof(buffer));
            for (int i = 0; i < is.gcount(); ++i) {
                if (buffer[i] == '\n') {
                    if (offset + i >= pos)
                        return fmt::format("line %i, col %i", line + 1, pos - linestart);
                    ++line;
                    linestart = offset + i;
                }
            }
            offset += (int) is.gcount();
        }
        return "byte offset " + std::to_string(pos);
    };

    if (!ret) /* There was a parser / file IO error */
        throw std::runtime_error(fmt::format("Error while parsing \"%s\": %s (at %s)", filepath, ret.description(), offset(ret.offset)));

    enum ETag {
        EScene,
        EFilm,
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

    constexpr frozen::unordered_map<frozen::string, ETag, 13> tags = {
        {"Scene", EScene},
        {"Film", EFilm},
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
        node = node.next_sibling();

    if (node.type() != pugi::node_element)
        throw std::runtime_error(
            fmt::format("Error while parsing \"%s\": unexpected content at %s", filepath, offset(node.offset_debug())));

    auto gettag = [&filepath, &offset, &tags](pugi::xml_node& node) {
        auto it = tags.find(frozen::string(node.name()));
        if (it == tags.end())
            throw std::runtime_error(fmt::format("Error while parsing \"%s\": unexpected tag \"%s\" at %s",
                filepath, node.name(), offset(node.offset_debug())));
        int tag = it->second;
        return tag;
    };

    // Make this part together with default value into a template
    // Parse the template before loading a scene file
    // Or we could just let user input invalid attributes and simply ignore them
    constexpr AttrSet<3> film_attributes = {
        "width",
        "height",
        "filename"
    };

    constexpr AttrSet<7> camera_attributes = {
        "resolution",
        "position",
        "lookat",
        "up",
        "fov",
        "near",
        "far"
    };

    constexpr AttrSet<2> sphere_attributes = {
        "radius",
        "center"
    };

    constexpr AttrSet<3> triangle_attributes = {
        "verta",
        "vertb",
        "vertc"
    };

    constexpr AttrSet<2> pointlight_attributes = {
        "spec",
        "position"
    };

    // TODO : The recursive node parsing in nori is more elegant..
    auto root_tag = gettag(node);
    if (root_tag == EScene) {
        // Found th root scene node, parse elements
        for (auto& ch : node.children()) {
            auto child_tag = gettag(ch);
            std::stringstream ss;
            // Now we assume all attribute component is float type..
            std::string type_str;
            switch (child_tag) {
                case EFilm:
                    for (auto& attr : node.attributes())
                        parse_attribute(ss, attr, film.get(), film_attributes);
                    break;

                case ECamera:
                    for (auto& attr : node.attributes())
                        parse_attribute(ss, attr, camera.get(), camera_attributes);
                    break;

                case EAccelerator:
                    break;

                case EObjects:
                    for (auto& object_node : node.children()) {
                        auto obj_tag = gettag(object_node);
                        if (obj_tag == ESphere) {
                            for (auto& attr : object_node.attributes()) {
                                auto obj_ptr = std::make_unique<Sphere>();
                                parse_attribute(ss, attr, obj_ptr.get(), sphere_attributes);
                                objects.push_back(std::move(obj_ptr));
                            }
                        }
                        else if (obj_tag == ETriangle) {
                            for (auto& attr : object_node.attributes()) {
                                auto obj_ptr = std::make_unique<Triangle>();
                                parse_attribute(ss, attr, obj_ptr.get(), triangle_attributes);
                                objects.push_back(std::move(obj_ptr));
                            }
                        }
                    }
                    break;

                case EMaterials:
                    for (auto& mat_node : node.children()) {    

                    }
                    break;

                case ELights:
                    for (auto& light_node : node.children()) {
                        auto light_tag = gettag(light_node);
                        if (light_tag == EPointLight) {
                            for (auto& attr : light_node.attributes()) {
                                auto lgt_ptr = std::make_unique<PointLight>();
                                parse_attribute(ss, attr, lgt_ptr.get(), pointlight_attributes);
                                lights.push_back(std::move(lgt_ptr));
                            }
                        }
                    }
                    break;

                default:
                    break;
            }
        }
    }
}