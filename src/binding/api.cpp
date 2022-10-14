#include "binding/utils.h"
#include "core/scene.h"

// Trampoline class for Integrator
template <typename BaseItgt=Integrator>
class PyIntegrator : public BaseItgt {
public:
    using BaseItgt::BaseItgt;

    void setup(Scene* scene) override {
        PYBIND11_OVERRIDE(
            void,
            BaseItgt,
            setup,
            scene
        );
    }

    RGBSpectrum Li(const Ray& r, const RecordContext* rctx) const override {
        PYBIND11_OVERRIDE_PURE(
            RGBSpectrum,
            BaseItgt,
            Li,
            r, rctx
        );
    }
};

template <typename BaseItgt>
class PyOSLIntegrator : public BaseItgt {
public:
    using BaseItgt::BaseItgt;

    void setup(Scene* scene) override {
        PYBIND11_OVERRIDE(
            void,
            BaseItgt,
            setup,
            scene
        );
    }

    RGBSpectrum Li(const Ray& r, const RecordContext* rctx) const override {
        PYBIND11_OVERRIDE(
            RGBSpectrum,
            BaseItgt,
            Li,
            r, rctx
        );
    }
};

void bind_api(py::module_& m) {
    py::module api = m.def_submodule("api",
        "Classes and functions exposed as APIs");

    // Integrator related
    py::class_<Integrator, PyIntegrator<>> itgt(api, "Integrator");
    itgt.def(py::init<>())
        .def("Li", &Integrator::Li)
        .def("get_random_light", &Integrator::get_random_light);

    py::class_<NormalIntegrator, Integrator, PyIntegrator<NormalIntegrator>> nml_itgt(api, "NormalIntegrator");
    nml_itgt.def(py::init<>())
            .def("Li", &NormalIntegrator::Li);

    py::class_<AmbientOcclusionIntegrator, Integrator, PyIntegrator<AmbientOcclusionIntegrator>> ao_itgt(api, "AmbientOcclusionIntegrator");
    ao_itgt.def(py::init<>())
           .def("Li", &AmbientOcclusionIntegrator::Li);

    py::class_<OSLBasedIntegrator, Integrator, PyIntegrator<OSLBasedIntegrator>> osl_itgt(api, "OSLBasedIntegrator");
    osl_itgt.def(py::init<>());

    py::class_<WhittedIntegrator, OSLBasedIntegrator, PyOSLIntegrator<WhittedIntegrator>> wht_itgt(api, "WhittedIntegrator");
    wht_itgt.def(py::init<>())
            .def("Li", &WhittedIntegrator::Li);

    py::class_<PathMatsIntegrator, OSLBasedIntegrator, PyOSLIntegrator<PathMatsIntegrator>> mats_itgt(api, "PathMatsIntegrator");
    mats_itgt.def(py::init<>())
             .def("Li", &PathMatsIntegrator::Li);

    py::class_<PathEmsIntegrator, OSLBasedIntegrator, PyOSLIntegrator<PathEmsIntegrator>> ems_itgt(api, "PathEmsIntegrator");
    ems_itgt.def(py::init<>())
            .def("Li", &PathEmsIntegrator::Li);

    py::class_<PathIntegrator, OSLBasedIntegrator, PyOSLIntegrator<PathIntegrator>> path_itgt(api, "PathIntegrator");
    path_itgt.def(py::init<>())
             .def("Li", &PathIntegrator::Li);

    //Scene related
    py::class_<Scene> scene(api, "Scene");
    scene.def(py::init<>())
         .def("parse_from_file", &Scene::parse_from_file,
            "parse scene from a description file")
         .def("create_integrator", &Scene::create_integrator,
            "create integrator");
}