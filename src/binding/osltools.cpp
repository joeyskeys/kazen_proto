#include "binding/utils.h"
#include "shading/querier.h"

void bind_osltools(py::module_& m) {
    py::module ot = m.def_submodule("osl",
        "OSL related utilities");

    py::class_<Param> pacl(ot, "Param");
    pacl.def(py::init<>())
        .def_readwrite("isoutput", &Param::isoutput)
        .def_readwrite("validdefault", &Param::validdefault)
        .def_readwrite("varlenarray", &Param::varlenarray)
        .def_readwrite("isstruct", &Param::isstruct)
        .def_readwrite("isclosure", &Param::isclosure)
        .def("getname", &Param::getname)
        .def("gettype", &Param::gettype)
        .def("getbasetype", &Param::getbasetype)
        .def("getdefaulti", &Param::getdefaulti)
        .def("getdefaultf", &Param::getdefaultf)
        .def("getdefaults", &Param::getdefaults)
        .def("getstructname", &Param::getstructname)
        .def("getfields", &Param::getfields)
        .def("getmetadatas", &Param::getmetadatas);

    py::class_<Querier> pycl(ot, "Querier");
    pycl.def(py::init<>())
        .def(py::init<const std::string&, const std::string&>())
        .def("open", &Querier::open)
        .def("shadertype", &Querier::shadertype)
        .def("shadername", &Querier::shadername)
        .def("nparams", &Querier::nparams)
        .def("getparam", &Querier::getparam)
        .def("getparamname", &Querier::getparamname)
        .def("getparamtype", &Querier::getparamtype)
        .def("getparambasetype", &Querier::getparambasetype)
        .def("getdefaultsi", &Querier::getdefaultsi)
        .def("getdefaultsf", &Querier::getdefaultsf)
        .def("getdefaultss", &Querier::getdefaultss);
}