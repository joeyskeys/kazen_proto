#include <pybind11/pybind11.h>

#include "base/vec.h"
#include "shading/microfacet.h"

namespace py = pybind11;

template <typename Dist>
py::class_<Dist> bind_mdf(py::module& m, const char* name) {
    py::class_<Dist> pycl(m, name);

    pycl.def(py::init<const Vec3f&, const float, const float>())
        .def("sample_m", &Dist::sample_m, "sample a microfacet normal direction")
        .def("D", &Dist::D, "calculate the distribution function value")
        .def("pdf", &Dist::pdf, "calculate the pdf of the given direction")
        .def("G", &Dist::G, "calculate the geometric factor of the distribution")
        .def_readwrite("wi", &Dist::wi, "reversed viewing direction")
        .def_readwrite("xalpha", &Dist::xalpha, "roughness alpha alone x")
        .def_readwrite("yalpha", &Dist::yalpha, "roughenss alpha along y");

    return pycl;
}

void bind_microfacet(py::module_& m) {
    py::module mdf = m.def_submodule("mdf",
        "Microfacet related classes and methods");

    bind_mdf<MicrofacetInterface<GGXDist>>(mdf, "GGX");
    bind_mdf<MicrofacetInterface<BeckmannDist>>(mdf, "Beckmann");
}