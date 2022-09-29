#include <pybind11/pybind11.h>

namespace py = pybind11;

// Definitions of binding function in each file
void bind_microfacet(py::module_&);
void bind_utils(py::module_&);

PYBIND11_MODULE(pyzen, m) {
    bind_microfacet(m);
}