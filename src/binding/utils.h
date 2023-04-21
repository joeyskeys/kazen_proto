#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

inline py::module create_submodule(py::module& m, const char *name) {
    std::string full_name = std::string(PyModule_GetName(m.ptr())) + "." + name;
    py::module module = py::reinterpret_steal<py::module>(PyModule_New(full_name.c_str()));
    m.attr(name) = module;
    return module;
}