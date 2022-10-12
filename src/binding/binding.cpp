#include "binding/utils.h"

// Definitions of binding function in each file
void bind_basetypes(py::module_&);
void bind_microfacet(py::module_&);
void bind_sampler(py::module_&);

PYBIND11_MODULE(pyzen, m) {
    bind_basetypes(m);
    bind_microfacet(m);
    bind_sampler(m);
}