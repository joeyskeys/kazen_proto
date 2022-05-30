#pragma once

#include <filesystem>
#include <fstream>

#include "shading/compiler.h"

namespace fs = std::filesystem;

static std::string load_file(const fs::path& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);

    if (!file.good()) {
        std::cerr << "Failed open file : " << path << std::endl;
        return std::string();
    }

    size_t file_size = fs::file_size(path);
    std::string buffer(file_size, 0);
    file.seekg(0);
    file.read(buffer.data(), file_size);
    file.close();

    return buffer;
}

class Shader {
public:
    Shader(const fs::path& p) {
        m_source = load_file(p);
    }

    bool compile_shader(const ShaderCompiler* compiler);

    std::string m_source;
    std::string m_binary;
};