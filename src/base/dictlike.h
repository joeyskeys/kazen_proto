#pragma once

#include <string>
#include <functional>

using FuncPtr = std::function<void(void)>;

class DictLike {
public:
    virtual void* address_of(const std::string& name) {
        return nullptr;
    }
};

template <typename T>
struct TypeInfo {
    using type = T;
    constexpr static const char *name = "null";
    constexpr static const int  namelength = 4;
};

template <>
struct TypeInfo<float> {
    using type = float;
    constexpr static const char *name = "float";
    constexpr static const int  namelength = 5;
};

template <>
struct TypeInfo<int> {
    using type = int;
    constexpr static const char *name = "int";
    constexpr static const int  namelength = 3;
};