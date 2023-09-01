find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

set(SRC_FILES
    base_types.cpp
    binding.cpp
    bsdfs.cpp
    bssrdfs.cpp
    microfacet.cpp
    osltools.cpp
    sampler.cpp
    api.cpp)

pybind11_add_module(pyzen ${SRC_FILES})

target_include_directories(pyzen PRIVATE ${pybind11_INCLUDE_DIRS})
target_link_libraries(pyzen PRIVATE
    ${ASSIMP_LIBRARIES}
    ${TBB_LIBRARIES}
    ${OIIO_LIBRARIES}
    ${OCIO_LIBRARIES}
    ${OSL_LIBRARIES}
    ${Pugixml_LIBRARIES}
    ${EMBREE_LIBRARIES}
    kazen_proto_lib)

add_custom_command(TARGET pyzen POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:pyzen>
        ${CMAKE_SOURCE_DIR}/plugin/bazen)

add_custom_target(copylib ALL
    DEPENDS pyzen)

install(TARGETS pyzen DESTINATION ${CMAKE_INSTALL_PREFIX}/python)