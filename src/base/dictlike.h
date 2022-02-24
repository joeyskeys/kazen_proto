#pragma once

#include <string>
#include <functional>
#include <iostream>

#include <boost/hana.hpp>

namespace hana = boost::hana;

/* 
 * Boost Describe is also an interesting library to checkout which provide
 * similar functionality but have to cope with mp11. The interfaces it
 * provide are quite easy to use, checkout the source later
 */

using FuncPtr = std::function<void(void)>;

class DictLike {
public:
    virtual void* address_of(const std::string& name) {
        return nullptr;
    }
};

template <typename T>
class DictLikeT {
public:
    void* address_of(const std::string& searched_name) {
        void* ret = nullptr;
        hana::for_each(hana::accessors<T>(), hana::fuse([&](auto name, auto accessor) {
            auto runtime_str = std::string(name.c_str());
            auto& tmp = accessor(*reinterpret_cast<T*>(this));

            if (runtime_str == searched_name)
                ret = &tmp;
        }));

        return ret;
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