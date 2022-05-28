#pragma once

#include <filesystem>
#include <vector>

#include <OSL/oslcomp.h>

namespace fs = std::filesystem;

class ShaderCompiler {
public:
    fs::path                                m_stdosl_path;
    //std::unique_ptr<OSL::ErrorHandler>  m_error_handler;
    std::vector<std::string>                m_options;

    ShaderCompiler(const fs::path& stdosl_path)
        : m_stdosl_path(stdosl_path)
    {}

    void clear_options();
    void add_options(const std::string& option);
    bool compile_buffer(const std::string& source_code, std::string& ret) const;
};