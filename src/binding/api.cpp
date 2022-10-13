#include "binding/utils.h"
#include "core/scene.h"

void bind_api(py::module_& m) {
    //Scene related
    py::module api = m.def_submodule("api",
        "Classes and functions exposed as APIs");

    py::class_<Scene> scene(api, "Scene");
    scene.def(py::init<>())
         .def("parse_from_file", &Scene::parse_from_file,
            "parse scene from a description file")
         .def("create_integrator", &Scene::create_integrator,
            "create integrator");

    // Integrator related
    //py::class_<Integrator>
}