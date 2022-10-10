find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

set(SRC_FILES
    binding.cpp
    base_types.cpp
    microfacet.cpp)

pybind11_add_module(pyzen ${SRC_FILES})

target_include_directories(pyzen PRIVATE ${pybind11_INCLUDE_DIRS})