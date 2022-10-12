#include "binding/utils.h"
#include "core/sampler.h"

void bind_sampler(py::module_& m) {
    py::module spl = m.def_submodule("sampler",
        "Sampler related classes and methods");
    
    py::class_<Sampler> pycl(spl, "Sampler");
    pycl.def(py::init<>())
        .def("seed", &Sampler::seed)
        .def("randomf", &Sampler::randomf)
        .def("random2f", &Sampler::random2f)
        .def("random3f", &Sampler::random3f)
        .def("random4f", &Sampler::random4f)
        .def("randomi", &Sampler::randomi);
}