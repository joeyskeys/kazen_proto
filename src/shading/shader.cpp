#include "shader.h"

bool Shader::compile_shader(const ShaderCompiler* compiler) {
    if (!m_binary.empty())
        return true;

    if (m_source.empty()) {
        std::cerr << "No source code presented" << std::endl;
        return false;
    }

    if (!compiler) {
        std::cerr << "Compiler not presented" << std::endl;
        return false;
    }

    auto ok = compiler->compile_buffer(m_source, m_binary);
    if (!ok)
        std::cerr << "Compilation failed" << std::endl;
    
    return ok;
}