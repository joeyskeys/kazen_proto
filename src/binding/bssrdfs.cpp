#include "binding/utils.h"
#include "shading/bssrdfs.h"

template <typename BSSRDF>
py::class_<BSSRDF> bind_bssrdf(py::module& m, const char* name) {
    py::class_<BSSRDF> pycl(m, name);

    pycl.def_static("eval", &BSSRDF::eval, "evaluate the bssrdf")
        .def_static("sample", &BSSRDF::sample, "sample the bssrdf");

    return pycl;
}

void bind_bssrdfs(py::module_& m) {
    py::module bssrdfs = m.def_submodule("bssrdfs",
        "BSSRDF functions");

    bind_bssrdf<KpStandardDipole>(bssrdfs, "KpStandardDipole");
    bind_bssrdf<KpBetterDipole>(bssrdfs, "KpBetterDipole");
}