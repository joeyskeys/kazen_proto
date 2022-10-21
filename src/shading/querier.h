#pragma once

#include <OSL/oslquery.h>

// A wrapper classes for easy python binding
class Querier : public OSL::OSLQuery {
    Querier() : OSL::OSLQuery() {}
    Querier(const std::string& shadername, const std::string& searchpath)
        : OSL::OSLQuery(shadername, searchpath)
    {}

    inline bool open(const std::string& shadername, const std::string& searchpath) {
        return OSL::OSLQuery::open(shadername, searchpath);
    }
    inline std::string shadertype() const {
        return OSL::OSLQuery::shadertype();
    }

    inline std::string shadername() const {
        return OSL::OSLQuery::shadername();
    }

    inline size_t nparams() const {
        return OSL::OSLQuery::nparams();
    }

    inline std::string getparamname(size_t i) const {
        return OSL::OSLQuery::getparam(i)->name;
    }

    std::string getparamtype(size_t i) const;
};