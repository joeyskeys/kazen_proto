#include <OSL/oslconfig.h>

#include "base/vec.h"
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

class PyRenderCallback : public RenderCallback {
public:
    void on_tile_end(Film* film, uint32_t tile_id) override {
        PYBIND11_OVERRIDE(
            void,
            RenderCallback,
            on_tile_end,
            film,
            tile_id
        );
    }
    /*
    // This implementation is an example of customed impl, but this will cause
    // an weird crash...
    // Leave it as a comment here for now
    void on_tile_end(std::shared_ptr<Film>& film, uint32_t tile_id) override {
        pybind11::gil_scoped_acquire gil;
        pybind11::function py_func = pybind11::get_override(this, "on_tile_end");
        if (py_func) {
            std::cout << "found overrided python function" << std::endl;
            py_func(film, tile_id);
            std::cout << "finished running python function" << std::endl;
        }
    }
    */
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
         .def("build_bvh", &Scene::build_bvh,
            "build acceleration structure")
         .def("add_point_light", &Scene::add_point_light,
            "add a point light into the scene")
         .def("set_shader_search_path", &Scene::set_shader_search_path,
            "set shader search path for OSL shading system")
         .def("begin_shader_group", &Scene::begin_shader_group,
            "BeginShaderGroup call to OSL")
         .def("end_shader_group", &Scene::end_shader_group,
            "EndShaderGroup call to OSL")
         .def("load_oso_shader", &Scene::load_oso_shader,
            "load the built-in oso shader")
         .def("connect_shader", &Scene::connect_shader,
            "connect shader's attributes")
         .def("set_shader_param_bool", &Scene::set_shader_param<bool>,
            "set bool shader parameter value")
         .def("set_shader_param_float", &Scene::set_shader_param<float>,
            "set float shader parameter value")
         .def("set_shader_param_int", &Scene::set_shader_param<int>,
            "set float shader parameter value")
         .def("set_shader_param_vec3f", &Scene::set_shader_param<base::Vec3f>,
            "set vec3f shader parameter value")
         .def("set_shader_param_vec4f", &Scene::set_shader_param<base::Vec4f>,
            "set vec4f shader parameter value")
         .def("set_shader_param_str", &Scene::set_shader_param<std::string>,
            "set string shader parameter value")
         .def("set_shader_param_ustr", &Scene::set_shader_param<OSL::ustring>,
            "set ustring shader parameter value");

    py::enum_<AcceleratorType>(api, "AcceleratorType", "Accelerator type enum")
        .value("BVH", AcceleratorType::BVH)
        .value("Embree", AcceleratorType::Embree);

    py::enum_<IntegratorType>(api, "IntegratorType", "Integrator type enum")
        .value("NormalIntegrator", IntegratorType::NormalIntegrator)
        .value("AmbientOcclusionIntegrator", IntegratorType::AmbientOcclusionIntegrator)
        .value("WhittedIntegrator", IntegratorType::WhittedIntegrator)
        .value("PathMatIntegrator", IntegratorType::PathMatsIntegrator)
        .value("PathEmsIntegrator", IntegratorType::PathEmsIntegrator)
        .value("PathIntegrator", IntegratorType::PathIntegrator);

    // Renderer related
    py::class_<RenderCallback, PyRenderCallback> render_callback(api, "RenderCallback");
    render_callback.def(py::init<>())
                   .def("on_tile_end",
                        [](RenderCallback& cbk, Film* film, uint32_t tid) {
                            py::gil_scoped_acquire acquire_gil;
                            cbk.on_tile_end(film, tid);
                        }, "callback function called when tile finished render");

    py::class_<Renderer> renderer(api, "Renderer");
    renderer.def(py::init<>())
            .def(py::init<const uint, const uint, RenderCallback* const>())
            .def("render",
                [](Renderer& renderer, const std::string& scene_file, const std::string& output) {
                    py::gil_scoped_release release_gil;
                    renderer.render(scene_file, output);
                }, "start render")
            .def("render",
                [](Renderer& renderer, Scene& scene, const std::string& output) {
                    py::gil_scoped_release release_gil;
                    renderer.render(scene, output);
                }, "start render")
            .def("render",
                [](Renderer& renderer, Scene& scene, const uint32_t x, const uint32_t y) {
                    py::gil_scoped_release release_gil;
                    renderer.render(scene, x, y);
                }, "start debug render");
}