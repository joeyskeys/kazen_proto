#include "querier.h"

std::string Querier::getparamtype(size_t i) const {
    auto type = OSL::OSLQuery::getparam(i)->type;

    switch (type) {
        case OSL::TypeDesc::INT32:
            return "int";

        case OSL::TypeDesc::FLOAT:
            return "float";

        case OSL::TypeDesc::DOUBLE:
            return "double";

        case OSL::TypeDesc::STRING:
            return "string";

        case OSL::TypeDesc::VEC2:
            return "vec2";

        case OSL::TypeDesc::VEC3:
            return "vec3";

        case OSL::TypeDesc::VEC4:
            return "vec4";

        case OSL::TypeDesc::MATRIX33:
            return "mat3";

        case OSL::TypeDesc::MATRIX44:
            return "mat4";

        case OSL::TypeDesc::COLOR:
            return "color";

        case OSL::TypeDesc::POINT:
            return "point";

        case OSL::TypeDesc::VECTOR:
            return "vector";

        case OSL::TypeDesc::NORMAL:
            return "normal";

        default:
            return fmt::format("Unmapped type or unknown type : {}", type);
    }
}