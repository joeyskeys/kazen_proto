#include "binding/utils.h"
#include "shading/querier.h"

void bind_osltools(py::module_& m) {
    py::module ot = m.def_submodule("osl",
        "OSL related utilities");

    py::class_<Querier> pycl(ot, "Querier");
    pycl.def(py::init<>())
        .def(py::init<const std::string&, const std::string&>())
        .def("open", &Querier::open)
        .def("shadertype", &Querier::shadertype)
        .def("shadername", &Querier::shadername)
        .def("nparams", &Querier::nparams)
        .def("getparamname", &Querier::getparamname)
        .def("getparamtype", &Querier::getparamtype);
}