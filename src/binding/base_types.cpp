#include "binding/utils.h"
#include "base/vec.h"
#include "base/mat.h"

template <typename V>
py::class_<V> bind_vec(py::module& m, const char* name) {
    using T = V::Scalar;
    using N = V::Size;

    py::class_<V> pycl(m, name);

    pycl.def(py::init<>())
        .def(py::init<const T&>()
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self - py::self)
        .def(-py::self)
        .def(py::self -= py::self)
        .def(py::self * T())
        .def(py::self * py::self)
        .def(py::self *= T())
        .def(py::self *= py::self)
        .def(T() * py::self)
        .def(py::self / T())
        .def(py::self / py::self)
        .def(py::self /= T())
        .def(py::self /= py::self)
        .def(T() / py::self)
        .def(py::self == py::self)
        .def("__getitem__", [](const V& v, size_t idx) {
            if (idx >= N)
                throw py::index_error();
            return v[idx];
        })
        .def("__setitem__", [](const V& v, size_t idx) {
            if (idx >= N)
                throw py::index_error();
            return v[idx];
        })
        .def("dot", &V::dot)
        .def("cross", &V::cross)
        .def("is_zero", &V::is_zero)
        .def("normalize", &V::normalize)
        .def("normalized", &V::normalized)
        .def("length_squared", &V::length_squared)
        .def("length", &V::length)
        .def("sum", &V::sum)
        .def("abs", &V::abs)
        .def("exp", &V::exp)
        .def("max_component", &V::max_component)
        .def("min_component", &V::min_component)
        .def("__repr__", &V::to_str)
        .def_readwrite("x", &V::arr[0])
        .def_readwrite("y", &V::arr[1])
        .def_readwrite("r", &V::arr[0])
        .def_readwrite("g", &V::arr[1]);

    if constexpr (N > 2) {
        pycl.def_readwrite("z", &V::arr[2])
            .def_readwrite("b", &V::arr[2]);
    }

    if constexpr (N > 3) {
        pycl.def_readwrite("w", &V::arr[3]);
    }

    m.def("concat", &concat<T, N>, "concatnate a vector and a scalar")
     .def("max_component", &max_component<T, N>, "get the max component of a vector")
     .def("min_component", &min_component<T, N>, "get the min component of a vector")
     .def("sum", &sum<T, N>, "calculate the component sum of a vector")
     .def("length", &length<T, N>, "get the length of a vector")
     .def("length_squared", &length_squared<T, N>, "get the squared length of a vector")
     .def("dot", &dot<V, T, N>, "calculate the dot product of two vectors")
     .def("cross", &cross<T, N>, "calculate the cross product of two vectors")
     .def("normalize", &normalize<T, N>, "get the normalized vector")
     .def("abs", &abs<T, N>, "get the component wise absolute value of a vector")
     .def("exp", &exp<T, N>, "get the component wise exponential of a vector")
     .def("lerp", py::overload_cast<const V&, const V&, const T>(&lerp<T, N>),
        "lerp between two vectors by a scalar")
     .def("lerp", py::overload_cast<const V&, const V&, const V&>(&lerp<T, N>),
        "lerp between two vectors component wise by a vector")
     .def("is_zero", &is_zero<T, N>, "test if vector length is zero")
     .def("vec_min", &vec_min<T, N>, "get the minimum component of a vector")
     .def("vec_max", &vec_max<T, N>, "get the maximum component of a vector")
     .def("to_string", &to_string<T, N>, "stringtify vector");
}

void bind_basetypes(py::module_& m) {
    py::module sub = create_submodule(m, "math");

    sub.doc() = "Mathematical related stuffs";

    bind_vec<Vec2f>(m, "Vec2f");
    bind_vec<Vec2d>(m, "Vec2d");
    bind_vec<Vec2i>(m, "Vec2i");
    bind_vec<Vec3f>(m, "Vec3f");
    bind_vec<Vec3d>(m, "Vec3d");
    bind_vec<Vec3i>(m, "Vec3i");
    bind_vec<Vec4f>(m, "Vec4f");
    bind_vec<Vec4d>(m, "Vec4d");
    bind_vec<Vec4i>(m, "Vec4i");
}