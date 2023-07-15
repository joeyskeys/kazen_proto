#include "binding/utils.h"
#include "base/vec.h"
#include "base/mat.h"
#include "core/film.h"

/*
// Create a tampoline class like this will cause problem when pass arguments
// to APIs that takes the raw Vec object
// Leave this code here as a mark
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
*/

// The easiest way to create Vec from python types
template <typename T, base::vec_n_type_t N, typename P, typename =
    std::enable_if_t<std::is_same_v<P, py::tuple> || std::is_same_v<P, py::list>>>
base::Vec<T, N> create_vec(P& arr) {
    assert(arr.size() == N);
    base::Vec<T, N> ret;

    for (int i = 0; i < N; i++)
        ret[i] = arr[i].template cast<T>();

    return ret;
}

#define T typename V::Scalar
#define N V::Size

template <typename V>
void bind_create_vec_func(py::module& m, const char* name) {
    m.def(name, py::overload_cast<py::tuple&>(&create_vec<T, N, py::tuple>),
        "create a Vec from python tuple")
     .def(name, py::overload_cast<py::list&>(&create_vec<T, N, py::list>),
        "create a vec from python list");
}

// Math types
template <typename V>
py::class_<V> bind_vec(py::module& m, const char* name) {
    // Cannot use aliasing in a template function ???
    //using T = V::Scalar;
    //using N = V::Size;

    py::class_<V> pycl(m, name);

    pycl.def(py::init<>())
        .def(py::init<const T&>())
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
        m.def("cross", static_cast<T(*)(const V&, const V&)>(base::cross));
    }
    else if constexpr (N == 3) {
        m.def("cross", static_cast<V(*)(const V&, const V&)>(base::cross));
    }

    if constexpr (std::is_floating_point_v<T>) {
        m.def("lerp", py::overload_cast<const V&, const V&, const T>(&base::lerp<T, N>),
            "lerp between two vectors by a scalar")
        .def("lerp", py::overload_cast<const V&, const V&, const V&>(&base::lerp<T, N>),
            "lerp between two vectors component wise by a vector");
    }

    return pycl;
}

#undef T
#undef N

template <typename T, base::mat_n_type_t N, typename P>
base::Mat<T, N> create_mat(P& arr) {
    assert(arr.size() == N || arr.size() == N * N);
    base::Mat<T, N> ret;

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
            auto e = py::cast<P>(arr[i]);
            ret[i] = create_vec<T, N, P>(e);
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
            ret.data()[i] = arr[i].template cast<T>();
    }

    return ret;
}

#define T typename M::ValueType
#define V typename M::VecType
#define N M::dimension

template <typename M>
void bind_create_mat_func(py::module& m, const char* name) {
    m.def(name, py::overload_cast<py::tuple&>(&create_mat<T, N, py::tuple>),
        "create a Mat from python tuple")
     .def(name, py::overload_cast<py::list&>(&create_mat<T, N, py::list>),
        "create a Mat from python list");
}

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

    bind_vec<base::Vec2f>(vec, "Vec2f");
    bind_vec<base::Vec2d>(vec, "Vec2d");
    bind_vec<base::Vec2i>(vec, "Vec2i");
    bind_vec<base::Vec3f>(vec, "Vec3f");
    bind_vec<base::Vec3d>(vec, "Vec3d");
    bind_vec<base::Vec3i>(vec, "Vec3i");
    bind_vec<base::Vec4f>(vec, "Vec4f");
    bind_vec<base::Vec4d>(vec, "Vec4d");
    bind_vec<base::Vec4i>(vec, "Vec4i");

    // Since we cannot have element number of python list/tuple at compile
    // time, we have to create explicit create function...
    bind_create_vec_func<base::Vec2f>(vec, "create_vec2f");
    bind_create_vec_func<base::Vec2d>(vec, "create_vec2d");
    bind_create_vec_func<base::Vec2i>(vec, "create_vec2i");
    bind_create_vec_func<base::Vec3f>(vec, "create_vec3f");
    bind_create_vec_func<base::Vec3d>(vec, "create_vec3d");
    bind_create_vec_func<base::Vec3i>(vec, "create_vec3i");
    bind_create_vec_func<base::Vec4f>(vec, "create_vec4f");
    bind_create_vec_func<base::Vec4d>(vec, "create_vec4d");
    bind_create_vec_func<base::Vec4i>(vec, "create_vec4i");

    py::module mat = m.def_submodule("mat",
        "Matrix related classes and methods");

    bind_mat<base::Mat2f>(mat, "Mat2f");
    bind_mat<base::Mat2d>(mat, "Mat2d");
    bind_mat<base::Mat3f>(mat, "Mat3f");
    bind_mat<base::Mat3d>(mat, "Mat3d");
    bind_mat<base::Mat4f>(mat, "Mat4f");
    bind_mat<base::Mat4d>(mat, "Mat4d");

    bind_create_mat_func<base::Mat2f>(mat, "create_mat2f");
    bind_create_mat_func<base::Mat2d>(mat, "create_mat2d");
    bind_create_mat_func<base::Mat3f>(mat, "create_mat3f");
    bind_create_mat_func<base::Mat3d>(mat, "create_mat3d");
    bind_create_mat_func<base::Mat4f>(mat, "create_mat4f");
    bind_create_mat_func<base::Mat4d>(mat, "create_mat4d");

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
        .def("get_data", [](Tile& t) {
            std::vector<float> data(t.get_pixel_count() * 3);
            memcpy(data.data(), t.get_data_ptr<float>(), t.get_data_size());
            // Seems it's the array_t thing causing the halt, the detailed reason
            // is still not clear, keep the comments and code here just to remind.
            //return py::array_t<float>({t.get_pixel_count() * 3}, reinterpret_cast<float*>(data.get()));
            return data;
        })
        .def_readonly("x", &Tile::origin_x)
        .def_readonly("y", &Tile::origin_y)
        .def_readonly("width", &Tile::width)
        .def_readonly("height", &Tile::height);

    py::class_<Film> pyfl(concepts, "Film");
    pyfl.def(py::init<>())
        .def(py::init<uint, uint, uint, uint, const std::string&>())
        .def("generate_tiles", &Film::generate_tiles)
        .def("write_tiles", &Film::write_tiles)
        .def("set_film_color", &Film::set_film_color)
        .def("set_tile_color", &Film::set_tile_color)
        .def("get_tile_count", &Film::get_tile_count)
        .def_readonly("tiles", &Film::tiles);
}