#pragma once

#include <chrono>

inline auto get_time() {
    return std::chrono::high_resolution_clock::now();
}