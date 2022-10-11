#include "binding/utils.h"

// Definitions of binding function in each file
void bind_microfacet(py::module_&);
void bind_basetypes(py::module_&);

PYBIND11_MODULE(pyzen, m) {
    bind_basetypes(m);
    bind_microfacet(m);
}