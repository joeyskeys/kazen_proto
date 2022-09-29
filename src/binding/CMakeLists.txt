find_package(pybind11 REQUIRED)

set(SOURCE_FILS
    microfacet.cpp
    utils.cpp)

pybind11_add_module(pyzen ${SOURCE_FILES})
