find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

set(SRC_FILES
    binding.cpp
    base_types.cpp
    microfacet.cpp
    osltools.cpp
    sampler.cpp
    api.cpp
    ../core/accel.cpp
    ../core/camera.cpp
    ../core/film.cpp
    ../core/integrator.cpp
    ../core/light.cpp
    ../core/light_path.cpp
    ../core/renderer.cpp
    ../core/sampler.cpp
    ../core/scene.cpp
    ../core/shape.cpp
    ../shading/bsdf.cpp
    ../shading/bsdfs.cpp
    ../shading/compiler.cpp
    ../shading/microfacet.cpp
    ../shading/querier.cpp
    ../shading/renderservices.cpp
    ../shading/shader.cpp)

pybind11_add_module(pyzen ${SRC_FILES})

target_include_directories(pyzen PRIVATE ${pybind11_INCLUDE_DIRS})
target_link_libraries(pyzen PRIVATE
    ${ASSIMP_LIBRARIES}
    ${TBB_LIBRARIES}
    ${OIIO_LIBRARIES}
    ${OSL_LIBRARIES}
    ${Pugixml_LIBRARIES}
    ${EMBREE_LIBRARIES})

install(TARGETS pyzen DESTINATION ${CMAKE_INSTALL_PREFIX}/python)