
#include "compiler.h"

void ShaderCompiler::clear_options() {
    m_options.clear();
}

void ShaderCompiler::add_options(const std::string& option) {
    m_options.emplace_back(option);
}

bool ShaderCompiler::compile_buffer(
    const std::string& source_code,
    std::string& ret) const
{
    OSL::OSLCompiler compiler{};

    const bool ok =
        compiler.compile_buffer(
            source_code,
            ret,
            m_options,
            m_stdosl_path.c_str());
    return ok;
}