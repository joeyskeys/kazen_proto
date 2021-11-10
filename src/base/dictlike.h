#pragma once

#include <string>

class DictLike {
public:
    virtual void* address_of(const std::string& name) = 0;
};

template <typename T>
struct TypeInfo {
    using type = T;
    constexpr static const char *typename = "null";
    constexpr static const int  namelength = 4;
};

template <>
struct TypeInfo<float> {
    using type = float;
    constexpr static const char *typename = "float";
    constexpr static const int  namelength = 5;
};

template <>
struct TypeInfo<int> {
    using type = int;
    constexpr static const char *typename = "int";
    constexpr static const int  namelength = 3;
};