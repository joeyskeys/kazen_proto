#pragma once

#include <string>
#include <unordered_map>
#include <vector>

using PTXMap = std::unordered_map<std::string, std::tuple<std::string,
    std::string, void*>>;
