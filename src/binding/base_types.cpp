#include "binding/utils.h"
#include "base/vec.h"
#include "base/mat.h"
#include "core/film.h"


template <typename T, base::vec_n_type_t N>
class PyVec : public base::Vec<T, N> {
public:
    using Base = base::Vec<T, N>;

    // These ctors are ugly...
    // Is there a good way to forward ctors of base class directly to
    // here?
    PyVec() : Base() {}

    template <typename M, typename = std::enable_if_t<std::is_arithmetic_v<M>>>
    PyVec(const M& s) : Base(s) {}

    template <typename ...Ts, typename = std::enable_if_t<(... && std::is_arithmetic_v<Ts>)>>
    PyVec(Ts... args) : Base(args...) {}

    template <typename P, typename = std::enable_if_t<std::is_same_v<P, py::tuple> || std::is_same_v<P, py::list>>>
    PyVec(P& arr) {
        assert(arr.size() == N);
        for (int i = 0; i < N; i++)
            this->operator[](i) = arr[i].template cast<T>();
    }
};

template <typename T>
using PyVec2 = PyVec<T, 2>;

template <typename T>
using PyVec3 = PyVec<T, 3>;

template <typename T>
using PyVec4 = PyVec<T, 4>;

using PyVec2f = PyVec2<float>;
using PyVec2d = PyVec2<double>;
using PyVec2i = PyVec2<int>;
using PyVec3f = PyVec3<float>;
using PyVec3d = PyVec3<double>;
using PyVec3i = PyVec3<int>;
using PyVec4f = PyVec4<float>;
using PyVec4d = PyVec4<double>;
using PyVec4i = PyVec4<int>;

// Math types
template <typename V>
py::class_<V> bind_vec(py::module& m, const char* name) {
    // Cannot use aliasing in a template function ???
    //using T = V::Scalar;
    //using N = V::Size;

#define T typename V::Scalar
#define N V::Size

    py::class_<V> pycl(m, name);

    pycl.def(py::init<>())
        .def(py::init<const T&>())
        .def(py::init<py::tuple&>())
        .def(py::init<py::list&>())
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
        .def("x", py::overload_cast<>(&V::x, py::const_))
        .def("y", py::overload_cast<>(&V::y, py::const_))
        .def("r", py::overload_cast<>(&V::r, py::const_))
        .def("g", py::overload_cast<>(&V::g, py::const_));

    if constexpr (N == 2) {
        pycl.def(py::init<const T, const T>());
    }
    else if constexpr (N == 3) {
        pycl.def(py::init<const T, const T, const T>());
    }
    else {
        pycl.def(py::init<const T, const T, const T, const T>());
    }

    if constexpr (N > 2) {
        pycl.def("z", py::overload_cast<>(&V::z, py::const_))
            .def("b", py::overload_cast<>(&V::b, py::const_));
    }

    if constexpr (N > 3) {
        pycl.def("w", py::overload_cast<>(&V::w, py::const_));
    }

    if constexpr (N < 4) {
        pycl.def("cross", &V::cross);
    }

    m.def("concat", &base::concat<T, N>, "concatnate a vector and a scalar")
     .def("max_component", &base::max_component<T, N>, "get the max component of a vector")
     .def("min_component", &base::min_component<T, N>, "get the min component of a vector")
     .def("sum", &base::sum<T, N>, "calculate the component sum of a vector")
     .def("length", &base::length<T, N>, "get the length of a vector")
     .def("length_squared", &base::length_squared<T, N>, "get the squared length of a vector")
     .def("dot", &base::dot<T, N>, "calculate the dot product of two vectors")
     .def("normalize", &base::normalize<T, N>, "get the normalized vector")
     .def("abs", &base::abs<T, N>, "get the component wise absolute value of a vector")
     .def("exp", &base::exp<T, N>, "get the component wise exponential of a vector")
     .def("is_zero", &base::is_zero<T, N>, "test if vector length is zero")
     .def("vec_min", &base::vec_min<T, N>, "get the minimum component of a vector")
     .def("vec_max", &base::vec_max<T, N>, "get the maximum component of a vector")
     .def("to_string", &base::to_string<T, N>, "stringtify vector");

    // This is an interesting problem coz the template cross function may have 
    // different return type according to the N value
    // https://github.com/pybind/pybind11/issues/3174 this thread might be relavent
    // but decltype(auto) is tested and doesn't affect the result here
    // https://github.com/pybind/pybind11/issues/1153 this thread talks about the
    // overload_cast issue where return type cannot be deduced correctly and worth
    // a careful read
    // Currently explicit casting is used which is not elegant enough but it do 
    // solve the problem.
    /*
    if constexpr (N >= 2 && N < 4)
        m.def("cross", &base::cross<T, N>, "calculate the cross product of two vectors");
    */
    if constexpr (N == 2) {
        m.def("cross", static_cast<T(*)(const typename V::Base&,
            const typename V::Base&)>(base::cross));
    }
    else if constexpr (N == 3) {
        m.def("cross", static_cast<typename V::Base(*)(const typename V::Base&,
            const typename V::Base&)>(base::cross));
    }

    if constexpr (std::is_floating_point_v<T>) {
        m.def("lerp", py::overload_cast<const typename V::Base&,
            const typename V::Base&, const T>(&base::lerp<T, N>),
            "lerp between two vectors by a scalar")
        .def("lerp", py::overload_cast<const typename V::Base&,
            const typename V::Base&, const typename V::Base&>(&base::lerp<T, N>),
            "lerp between two vectors component wise by a vector");
    }

#undef T
#undef N

    return pycl;
}

template <typename T, base::mat_n_type_t N>
class PyMat : public base::Mat<T, N> {
public:
    using Base = base::Mat<T, N>;

    // Same problem here
    PyMat() : Base() {}

    template <typename ...Ts>
    PyMat(Ts... args) : Base(args...) {}

    template <typename P, typename = std::enable_if_t<std::is_same_v<P, py::tuple> || std::is_same_v<P, py::list>>>
    PyMat(P& arr) {
        assert(arr.size() == N || arr.size() == N * N);
        if (arr.size() == N) {
            // First loop to validate
            for (int i = 0; auto& e : arr) {
                if (!py::isinstance<py::tuple>(e) && !py::isinstance<py::list>(e)) {
                    throw std::runtime_error(
                        fmt::format("The {} element of the tuple/list is not tuple/list", i)
                    );
                }
                ++i;
            }

            // Second loop to assign elements
            for (int i = 0; i < N; i++) {
                auto e = py::cast<py::tuple>(arr[i]);
                this->operator[](i) = typename PyVec<T, N>::PyVec(e);
            }
        }
        else if (arr.size() == N * N) {
            // Again, validate first
            for (int i = 0; auto& e : arr) {
                if (!py::isinstance<py::float_>(e)) {
                    throw std::runtime_error(
                        fmt::format("The {} element of the tuple/list is not float", i)
                    );
                }
                ++i;
            }

            // Assign
            for (int i = 0; i < N * N; i++)
                this->data()[i] = arr[i].template cast<T>();
        }
    }
};

template <typename T>
using PyMat2 = PyMat<T, 2>;

template <typename T>
using PyMat3 = PyMat<T, 3>;

template <typename T>
using PyMat4 = PyMat<T, 4>;

using PyMat2f = PyMat2<float>;
using PyMat2d = PyMat2<double>;
using PyMat3f = PyMat3<float>;
using PyMat3d = PyMat3<double>;
using PyMat4f = PyMat4<float>;
using PyMat4d = PyMat4<double>;

#define T typename M::ValueType
#define V typename M::VecType
#define N M::dimension

// This piece of code is ugly... But currently seems no better way to do it.
// The core problems is I don't know any way to make template argument variadic
template <typename M>
inline void add_init_to_mat2(py::class_<M>& pycl) {
    pycl.def(py::init<const V&, const V&>())
        .def(py::init<const T, const T, const T, const T>());
}

template <typename M>
inline void add_init_to_mat3(py::class_<M>& pycl) {
    pycl.def(py::init<const V&, const V&, const V&>())
        .def(py::init<const T, const T, const T,
            const T, const T, const T,
            const T, const T, const T>());
}

template <typename M>
inline void add_init_to_mat4(py::class_<M>& pycl) {
    pycl.def(py::init<const V&, const V&, const V&, const V&>())
        .def(py::init<const T, const T, const T, const T,
            const T, const T, const T, const T,
            const T, const T, const T, const T,
            const T, const T, const T, const T>());
}

template <typename M>
py::class_<M> bind_mat(py::module& m, const char* name) {
    py::class_<M> pycl(m, name);

    pycl.def(py::init<>())
        .def(py::init<py::tuple&>())
        .def(py::init<py::list&>())
        .def(py::self * T())
        .def(py::self *= T())
        .def(py::self * base::Vec<T, N>())
        .def(py::self * py::self)
        .def(py::self *= py::self)
        .def("inverse", &M::inverse)
        .def("transpose", &M::transpose)
        .def_static("identity", &M::identity)
        .def_static("scale", &M::scale);

    if constexpr (N == 2)
        add_init_to_mat2<M>(pycl);
    else if constexpr (N == 3)
        add_init_to_mat3<M>(pycl);
    else if constexpr (N == 4)
        add_init_to_mat4<M>(pycl);

    if constexpr (N > 2) {
        pycl.def(py::self * base::Vec<T, N - 1>())
            .def_static("translate", &M::translate);
    }

    m.def("identity", &base::identity<T, N>, "create an identity matrix")
     .def("transpose", &base::transpose<T, N>, "return a transposed matrix")
     .def("inverse", &base::inverse<T, N>, "return a inversed matrix");
    
    if constexpr (N > 2) {
        m.def("translate", &base::translate<T, N>, "return the translation matrix")
         .def("scale", &base::scale<T, N>, "return the scale matrix");
    }
    if constexpr (N > 3) {
        m.def("rotate", &base::rotate<T, N>, "return the rotation matrix");
    }
    
    return pycl;
}

#undef T
#undef N

void bind_basetypes(py::module_& m) {
    // Math types
    py::module vec = m.def_submodule("vec",
        "Vector related classes and methods");

    bind_vec<PyVec2f>(vec, "Vec2f");
    bind_vec<PyVec2d>(vec, "Vec2d");
    bind_vec<PyVec2i>(vec, "Vec2i");
    bind_vec<PyVec3f>(vec, "Vec3f");
    bind_vec<PyVec3d>(vec, "Vec3d");
    bind_vec<PyVec3i>(vec, "Vec3i");
    bind_vec<PyVec4f>(vec, "Vec4f");
    bind_vec<PyVec4d>(vec, "Vec4d");
    bind_vec<PyVec4i>(vec, "Vec4i");

    py::module mat = m.def_submodule("mat",
        "Matrix related classes and methods");

    bind_mat<PyMat2f>(mat, "Mat2f");
    bind_mat<PyMat2d>(mat, "Mat2d");
    bind_mat<PyMat3f>(mat, "Mat3f");
    bind_mat<PyMat3d>(mat, "Mat3d");
    bind_mat<PyMat4f>(mat, "Mat4f");
    bind_mat<PyMat4d>(mat, "Mat4d");

    // Interesting and worth noting:
    // Explicitly instantiated function assigned to a variable is deduced as a
    // function pointer, e.g. translate3f itself is already function pointer thus
    // no need for the & operator
    mat.def("translate3f", base::translate3f,
        "return the translation matrix for 3D space")
       .def("rotate3f", base::rotate3f,
        "return the rotation matrix for 3D space")
       .def("scale3f", base::scale3f,
        "return the scale matrix for 3D space");

    // Concept types
    py::module concepts = m.def_submodule("concepts",
        "Concept types");
    
    py::class_<Tile> pytl(concepts, "Tile");
    pytl.def(py::init<uint, uint, uint, uint>())
        .def("set_pixel_color", &Tile::set_pixel_color)
        .def("set_tile_color", &Tile::set_tile_color)
        .def("get_data", &Tile::get_data_ptr<float>)
        .def_readonly("x", &Tile::origin_x)
        .def_readonly("y", &Tile::origin_y)
        .def_readonly("width", &Tile::width)
        .def_readonly("height", &Tile::height);

    py::class_<Film> pyfl(concepts, "Film");
    pyfl.def(py::init<>())
        .def(py::init<uint, uint, const std::string&>())
        .def("generate_tiles", &Film::generate_tiles)
        .def("write_tiles", &Film::write_tiles)
        .def("set_film_color", &Film::set_film_color)
        .def("set_tile_color", &Film::set_tile_color)
        .def("get_tile_count", &Film::get_tile_count)
        .def_readonly("tiles", &Film::tiles);
}