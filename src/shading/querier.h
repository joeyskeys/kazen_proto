#pragma once

#include <algorithm>

#include <OSL/oslquery.h>

// Wrapper classes for easy python binding
class Param : public OSL::OSLQuery::Parameter {
public:
    Param() : OSL::OSLQuery::Parameter() {}
    Param(const OSL::OSLQuery::Parameter& p) : OSL::OSLQuery::Parameter(p) {}

    inline std::string getname() const {
        return static_cast<std::string>(name);
    }

    inline std::string gettype() const {
        return type.c_str();
    }

    inline std::string getbasetype() const {
        return type.scalartype().c_str();
    }

    inline std::vector<int> getdefaulti() const {
        return idefault;
    }

    inline std::vector<float> getdefaultf() const {
        return fdefault;
    }

    inline std::vector<std::string> getdefaults() const {
        std::vector<std::string> ret;
        for (auto& us : sdefault) {
            ret.push_back(us.data());
        }
        return ret;
    }

    inline std::string getstructname() const {
        return static_cast<std::string>(structname);
    }

    inline std::vector<std::string> getfields() const {
        std::vector<std::string> ret;
        for (auto& us : fields) {
            ret.push_back(us.data());
        }
        return ret;
    }

    inline std::vector<Param> getmetadatas() const {
        std::vector<Param> ret;
        for (auto& m : metadata)
            ret.push_back(m);
        return ret;
    }
};

class Querier : public OSL::OSLQuery {
public:
    Querier() : OSL::OSLQuery() {}
    Querier(const std::string& shadername, const std::string& searchpath)
        : OSL::OSLQuery(shadername, searchpath)
    {}

    inline bool open(const std::string& shadername, const std::string& searchpath) {
        return OSL::OSLQuery::open(shadername, searchpath);
    }
    inline std::string shadertype() const {
        return OSL::OSLQuery::shadertype().data();
    }

    inline std::string shadername() const {
        return OSL::OSLQuery::shadername().data();
    }

    inline size_t nparams() const {
        return OSL::OSLQuery::nparams();
    }

    inline Param getparam(size_t i) const {
        return *OSL::OSLQuery::getparam(i);
    }

    inline std::string getparamname(size_t i) const {
        return OSL::OSLQuery::getparam(i)->name.data();
    }

    inline std::string getparamtype(size_t i) const {
        return OSL::OSLQuery::getparam(i)->type.c_str();
    }

    inline std::string getparambasetype(size_t i) const {
        return OSL::OSLQuery::getparam(i)->type.scalartype().c_str();
    }

    inline std::vector<int> getdefaultsi(size_t i) const {
        return OSL::OSLQuery::getparam(i)->idefault;
    }

    inline std::vector<float> getdefaultsf(size_t i) const {
        return OSL::OSLQuery::getparam(i)->fdefault;
    }

    inline std::vector<std::string> getdefaultss(size_t i) const {
        std::vector<std::string> ret;
        auto ss = OSL::OSLQuery::getparam(i)->sdefault;
        std::for_each(ss.begin(), ss.end(), [&ret](const auto& us) {
            ret.push_back(us.data());
        });
        return ret;
    }
};