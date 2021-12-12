#include <cstring>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

#include <boost/algorithm/string.hpp>
#include <fmt/core.h>
#include <frozen/set.h>
#include <frozen/string.h>
#include <frozen/unordered_map.h>
#include <pugixml.hpp>

#include "scene.h"

enum ETag {
    EScene,
    EFilm,
    ECamera,
    EAccelerator,
    EIntegrator,
    // objects
    EObjects,
    ESphere,
    ETriangle,
    // materials
    EMaterials,
    EShaderGroup,
    EShader,
    EParameter,
    EConnectShaders,
    // lights
    ELights,
    EPointLight,
    EInvalid
};

constexpr static frozen::unordered_map<frozen::string, ETag, 15> tags = {
    {"Scene", EScene},
    {"Film", EFilm},
    {"Camera", ECamera},
    {"Accelerator", EAccelerator},
    {"Integrator", EIntegrator},
    {"Objects", EObjects},
    {"Sphere", ESphere},
    {"Triangle", ETriangle},
    {"Materials", EMaterials},
    {"ShaderGroup", EShaderGroup},
    {"Shader", EShader},
    {"Parameter", EParameter},
    {"ConnectShaders", EConnectShaders},
    {"Lights", ELights},
    {"PointLight", EPointLight}
};

enum EType {
    EFloat,
    EInt,
    EVec3f,
    EStr
};

constexpr static frozen::unordered_map<frozen::string, EType, 4> types = {
    {"float", EFloat},
    {"int", EInt},
    {"float3", EVec3f},
    {"string", EStr}
};

template <int N>
using AttrSet = frozen::set<frozen::string, N>;

template <typename T>
inline T string_to(const std::string& s) {
    if constexpr (std::is_integral_v<T> && sizeof(T) == 4)
        return std::stoi(s);
    else if constexpr (std::is_floating_point_v<T>) {
        if constexpr (sizeof(T) == 4)
            return std::stof(s);
        else
            return std::stod(s);
    }
}

template <typename T>
//bool parse_components(const std::string& attrstr, std::stringstream& ss, void* dst) {
bool parse_components(const std::vector<std::string>& strs, void* dst) {
    assert(dst != nullptr);

    if constexpr (std::is_arithmetic_v<T>) {
        auto found = strs[0].find(TypeInfo<T>::name);
        if (found != std::string::npos) {
            int namelength = TypeInfo<T>::namelength;
            int cnt = 1;
            if (strs[0].size() > namelength)
                cnt = strs[0].back() - '0';

            // Stringstream is causing problem, use boost split
            // Assume we have 4 components at most
            auto buf = reinterpret_cast<T*>(dst);
            for (int i = 0; i < cnt; i++)
                buf[i] = string_to<T>(strs[i + 1]);
            return true;
        }
    }
    else if constexpr (std::is_same_v<std::string, std::decay_t<T>>) {
        auto strdst = reinterpret_cast<std::string*>(dst);
        *strdst = strs[1];
        return true;
    }
    
    return false;
}

template <typename T>
void parse_attribute(std::stringstream& ss, pugi::xml_attribute attr, DictLike* obj, T& attr_set) {
    auto it = attr_set.find(frozen::string(attr.name()));

    std::vector<std::string> ret;
    if (it != attr_set.end()) {
        // attribute string format : type value1 value2..
        boost::split(ret, attr.value(), boost::is_any_of(" "));

        // float  type
        if (parse_components<float>(ret, obj->address_of(attr.name())))
            return;
        // int type
        if (parse_components<int>(ret, obj->address_of(attr.name())))
            return;
    }
}

using Param = std::variant<std::string, int, float, double, Vec3f>;
using Params = std::unordered_map<std::string, std::pair<OSL::TypeDesc, Param>>;

template <typename T>
inline Param parse_attribute(std::stringstream& ss) {
    T tmp;
    ss >> tmp;
    return Param(tmp);
}

Params parse_attributes(pugi::xml_node node) {
    Params params;
    for (auto& child : node.children()) {
        for (auto& attr : child.attributes()) {
            std::stringstream ss;
            ss.str(attr.value());
            std::string typestr;
            ss >> typestr;

            auto type_pair = types.find(frozen::string(typestr));
            if (type_pair == types.end())
                continue;

            auto type_tag = type_pair->second;
            switch (type_tag) {
                case EFloat:
                    params.emplace(attr.name(), std::make_pair(OSL::TypeDesc::TypeFloat, parse_attribute<float>(ss)));
                    break;

                case EInt:
                    params.emplace(attr.name(), std::make_pair(OSL::TypeDesc::TypeInt, parse_attribute<int>(ss)));
                    break;

                case EVec3f:
                    params.emplace(attr.name(), std::make_pair(OSL::TypeDesc::TypeVector, parse_attribute<Vec3f>(ss)));
                    break;

                case EStr:
                    params.emplace(attr.name(), std::make_pair(OSL::TypeDesc::TypeString, parse_attribute<std::string>(ss)));
                    break;

                default:
                    break;
            }
        }
    }

    return params;
}

Scene::Scene()
    : film(std::make_unique<Film>())
    , camera(std::make_unique<Camera>())
    , accelerator(nullptr)
    , shadingsys(std::make_unique<OSL::ShadingSystem>(&rend, nullptr, &errhandler))
{
    camera->film = film.get();
    register_closures(shadingsys.get());
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
                        return fmt::format("line {}, col {}", line + 1, pos - linestart);
                    ++line;
                    linestart = offset + i;
                }
            }
            offset += (int) is.gcount();
        }
        return "byte offset " + std::to_string(pos);
    };

    if (!ret) /* There was a parser / file IO error */
        throw std::runtime_error(fmt::format("Error while parsing \"{}\": {} (at {})", filepath, ret.description(), offset(ret.offset)));


    auto& node = *doc.begin();
    // Skip over comments
    while (node.type() == pugi::node_comment || node.type() == pugi::node_declaration)
        node = node.next_sibling();

    if (node.type() != pugi::node_element)
        throw std::runtime_error(
            fmt::format("Error while parsing \"{}\": unexpected content at {}", filepath, offset(node.offset_debug())));

    auto gettag = [&filepath, &offset](pugi::xml_node& node) {
        auto it = tags.find(frozen::string(node.name()));
        if (it == tags.end())
            throw std::runtime_error(fmt::format("Error while parsing {}: unexpected tag \"{}\" at {}",
                filepath, node.name(), offset(node.offset_debug())));
        auto tag = it->second;
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

    constexpr AttrSet<3> sphere_attributes = {
        "radius",
        "center",
        "shader_name"
    };

    constexpr AttrSet<4> triangle_attributes = {
        "verta",
        "vertb",
        "vertc",
        "shader_name"
    };

    constexpr AttrSet<2> pointlight_attributes = {
        "spec",
        "position"
    };

    // TODO : The recursive node parsing in nori is more elegant..
    auto root_tag = gettag(node);
    if (root_tag == EScene) {
        // Found the root scene node, parse elements
        for (auto& sub_1 : node.children()) {
            auto child_tag = gettag(sub_1);
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
                    // BVH only for now
                    //accelerator = std::make_unique<BVHAccel>(objects, 0, objects.size());
                    accelerator = std::make_unique<BVHAccel>();
                    break;

                case EIntegrator:
                    // pt only for now
                    integrator = std::make_unique<Integrator>(camera.get(), film.get());
                    integrator->shadingsys = shadingsys.get();
                    integrator->shaders = &shaders;
                    break;

                case EObjects:
                    std::cout << "Parsing objects :" << std::endl;
                    for (auto& object_node : sub_1.children()) {
                        auto obj_tag = gettag(object_node);
                        if (obj_tag == ESphere) {
                            std::cout << "Parsing sphere object.." << std::endl;
                            for (auto& attr : object_node.attributes()) {
                                auto obj_ptr = std::make_shared<Sphere>();
                                parse_attribute(ss, attr, obj_ptr.get(), sphere_attributes);
                                objects.push_back(std::move(obj_ptr));
                            }
                        }
                        else if (obj_tag == ETriangle) {
                            std::cout << "Parsing triangle object.." << std::endl;
                            for (auto& attr : object_node.attributes()) {
                                auto obj_ptr = std::make_shared<Triangle>();
                                parse_attribute(ss, attr, obj_ptr.get(), triangle_attributes);
                                objects.push_back(std::move(obj_ptr));
                            }
                        }
                    }
                    break;

                case EMaterials:
                    for (auto& mat_node : sub_1.children()) {    
                        auto mat_tag = gettag(mat_node);
                        if (mat_tag == EShaderGroup) {
                            auto name_attr = mat_node.attribute("name");
                            if (!name_attr)
                                throw std::runtime_error(fmt::format("material doesn't have a name at {}", offset(mat_node.offset_debug())));

                            // Currently we ignore shadertype and commands
                            OSL::ShaderGroupRef shader_group = shadingsys->ShaderGroupBegin(name_attr.value());

                            for (auto& sub_node : mat_node.children()) {
                                auto sub_tag = gettag(sub_node);
                                if (sub_tag == EShader) {
                                    // We need a general purpose attribute parser here
                                    // Considering using visit pattern with variant

                                    auto params = parse_attributes(sub_node);
                                    for (const auto& [name, param_pair] : params) {
                                        // Integrator is not initialized..
                                        // Consider put shadingsys into Scene directly and
                                        // keep a pointer to it in the integrator
                                        shadingsys->Parameter(*shader_group, name.c_str(),
                                            param_pair.first, &param_pair.second);
                                    }

                                    auto type_attr = sub_node.attribute("type");
                                    auto name_attr = sub_node.attribute("name");
                                    auto layer_attr = sub_node.attribute("layer");
                                    const char* type = type_attr ? type_attr.value() : "surface";
                                    if (!name_attr || !layer_attr)
                                        throw std::runtime_error(fmt::format("Name or layer attribute not specified at {}", offset(sub_node.offset_debug())));
                                    shadingsys->Shader(*shader_group, type, name_attr.value(),
                                        layer_attr.value());
                                }
                                else if (sub_tag == EConnectShaders) {
                                    // integrator->shadingsys->ConnectShaders()
                                    auto sl = sub_node.attribute("srclayer");
                                    auto sp = sub_node.attribute("srcparam");
                                    auto dl = sub_node.attribute("dstlayer");
                                    auto dp = sub_node.attribute("dstparam");
                                    if (sl && sp && dl && dp)
                                        shadingsys->ConnectShaders(sl.value(), sp.value(),
                                            dl.value(), dp.value());
                                }
                            }
                        }
                    }
                    break;

                case ELights:
                    for (auto& light_node : sub_1.children()) {
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

    // Construct acceleration structure after all data is parsed
    auto bvh_ptr = reinterpret_cast<BVHAccel*>(accelerator.get());
    bvh_ptr->reset(objects, 0, objects.size());
}