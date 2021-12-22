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
    EStr,
    EFuncTrans
};

constexpr static frozen::unordered_map<frozen::string, EType, 5> types = {
    {"float", EFloat},
    {"int", EInt},
    {"float3", EVec3f},
    {"string", EStr},
    {"func_translate", EFuncTrans}
};

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

namespace {
using Comps = std::vector<std::string>;
using Param = std::variant<std::string, int, float, double, Vec3f>;
using Params = std::unordered_map<std::string, std::pair<OSL::TypeDesc, Param>>;
};

OSL::TypeDesc parse_attribute(const pugi::xml_attribute& attr, void* dst) {
    Comps comps;
    boost::split(comps, attr.value(), boost::is_any_of(" "));
    if (type_pair == types.end()) {
        std::cout << "Unknown type specified : " << std::endl;
        return;
    }

    OSL::TypeDesc ret;
    auto type_tag = type_pair->second;
    T* typed_dst = reinterpret_cast<T*>(dst);
    switch (type_tag) {
        case EFloat: {
            ret = OSL::TypeDesc::TypeFloat;
            *typed_dst = string_to<T>(comps[1]);
            break;
        }

        case EInt: {
            ret = OSL::TypeDesc::TypeInt;
            *typed_dst = string_to<T>(comps[1]);
            break;
        }

        case EVec3f: {
            ret = OSL::TypeDesc::TypeVector;
            for (int i = 0; i < T::dimension; i++)
                (*typed_dst)[i] = string_to<typename T::ValueType>(comps[i + 1]);
            break;
        }

        case EStr: {
            ret = OSL::TypeDesc::TypeString;
            *typed_dst = comps[1];
            break;
        }

        case EFuncTrans: {
            Vec3f trans{};
            parse_attribute<Vec3f>(ret, &trans);
            auto hitable = reinterpret_cast<HitablePtr>(dst);
            hitable->translate(trans);
            break;
        }

        default:
            // Never gonna happen
            break;
    }

    return ret;
}

inline std::pair<OSL::TypeDesc, Param> parse_attribute(const pugi::xml_attribute& attr) {
    Param ret;
    auto osl_type = parse_attribute(attr, &ret);
    return std::make_pair(osl_type, ret);
}

void parse_attributes(const pugi::xml_node& node, DictLike* obj) {
    for (auto& attr : node.attributes()) {
        void* dst = obj->address_of(attr.name());
        if (dst == nullptr) {
            std::cout << "Attribute " << attr.name() << "not used.." << std::endl;
            continue;
        }

        parse_attribute(attr, dst);
    }
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

    // Use a stack to store unprocessed node in order to recursively
    // process nodes
    std::vector<pugi::xml_node*> nodes_to_process;
    nodes_to_process.push_back(&node);
    OSL::ShaderGroupRef current_shader_group;
    while (nodes_to_process.size() > 0) {
        auto nodeptr = nodes_to_process.back();
        nodes_to_process.pop_back();
        auto tag = gettag(*nodeptr);
        switch (tag) {
            case EScene:
                break;

            case EFilm:
                parse_attributes(*nodeptr, film.get());
                break;

            case ECamera:
                parse_attributes(*nodeptr, camera.get());
                break;

            case EAccelerator:
                // BVH only for now
                accelerator = std::make_unique<BVHAccel>();
                break;

            case EIntegrator: {
                // pt only for now
                integrator = std::make_unique<Integrator>(camera.get(), film.get());
                integrator->shadingsys = shadingsys.get();
                integrator->shaders = &shaders;
                break;
            }

            case EObjects:
                break;

            case ESphere: {
                auto obj_ptr = std::make_shared<Sphere>();
                parse_attributes(*nodeptr, obj_ptr.get());
                objects.push_back(obj_ptr);
                break;
            }

            case ETriangle: {
                auto obj_ptr = std::make_shared<Triangle>();
                parse_attributes(*nodeptr, obj_ptr.get());
                objects.push_back(obj_ptr);
                break;
            }

            case EMaterials:
                break;

            case EShaderGroup: {
                auto name_attr = nodeptr->attribute("name");
                if (!name_attr)
                    throw std::runtime_error(fmt::format("No name specified for shader group at {}",
                        offset(nodeptr->offset_debug())));
                current_shader_group = shadingsys->ShaderGroupBegin(name_attr.value());
                shaders[name_attr.value()] = current_shader_group;
                break;
            }

            case EShader:
                // Kinda complicate here..
                break;

            case EParameter:
                // Only one attribute allowed
                auto attr = nodeptr->first_attribute();
                if (!attr)
                    throw std::runtime_error("Cannot find attribute specified in Parameter node");
                auto [osl_type, param] = parse_attribute(attr);
                shadingsys->Parameter(*current_shader_group, attr.name(), osl_type,
                    &param);
                break;

            case EConnectShaders: {
                // Ugly code for now..
                auto sl = nodeptr->attribute("srclayer");
                auto sp = nodeptr->attribute("srcparam");
                auto dl = nodeptr->attribute("dstlayer");
                auto dp = nodeptr->attribute("dstparam");
                if (sl && sp && dl && dp)
                    shadingsys->ConnectShaders(sl.value(), sp.value(),
                        dl.value(), dp.value());
                break;
            }

            case ELights:
                break;

            case EPointLight:
                auto lgt_ptr = std::make_unique<PointLight>();
                parse_attributes(*nodeptr, lgt_ptr.get());
                lights.emplace_back(std::move(lgt_ptr));
                break;

            case EInvalid:
                break;

            default:
                break;
        }

        for (auto& child : nodeptr->children())
            nodes_to_process.push_back(&child);
    }
}