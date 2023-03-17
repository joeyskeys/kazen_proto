#include "binding/utils.h"
#include "core/renderer.h"

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

template <typename CBK>
class PyRenderCallback : public CBK {
public:
    void on_tile_end() override {
        PYBIND11_OVERRIDE(
            void,
            CBK,
            on_tile_end,
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

    // Scene related
    py::class_<Scene> scene(api, "Scene");
    scene.def(py::init<>())
         .def("parse_from_file", &Scene::parse_from_file,
            "parse scene from a description file")
         .def("create_integrator", &Scene::create_integrator,
            "create integrator")
         .def("set_film", &Scene::set_film,
            "set film for the scene")
         .def("set_camera", &Scene::set_camera,
            "set camera for the scene")
         .def("set_accelerator", &Scene::set_accelerator,
            "set accelerator for the scene")
         .def("set_integrator", &Scene::set_integrator,
            "set integrator for the scene")
         .def("add_sphere", &Scene::add_sphere,
            "add a sphere into the scene")
         .def("add_triangle", &Scene::add_triangle,
            "add a triangle into the scene")
         .def("add_quad", &Scene::add_quad,
            "add a quad into the scene")
         .def("add_mesh", &Scene::add_mesh,
            "add a mesh into the scene")
         .def("add_point_light", &Scene::add_point_light,
            "add a point light into the scene");

    // Renderer related
    py::class_<RenderCallback> render_callback(api, "RenderCallback");
    render_callback.def(py::init<>())
                   .def("on_tile_end", &RenderCallback::on_tile_end,
                        "callback function called when tile finished render");

    py::class_<Renderer> renderer(api, "Renderer");
    renderer.def(py::init<>())
            .def(py::init<const uint, const uint>())
            .def("load_scene", &Renderer::load_scene,
                "load scene file from given path")
            .def("render", &Renderer::render,
                "start render");
}