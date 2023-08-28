cmake_minimum_required(VERSION 3.12)

project(kazen-prototype LANGUAGES CXX C CUDA)

cmake_policy(SET CMP0074 NEW)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "set build type")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)

list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/cmake"
    "${CMAKE_SOURCE_DIR}/cmake/modules")

option(USE_TBB "Whether to use TBB to enable multithreading" ON)

set(VECTORIZATION "None" CACHE STRING "Chooose vectorization solution")
set_property(CACHE VECTORIZATION PROPERTY STRINGS None Enoki Eigen)
set(USE_ENOKI OFF)
set(USE_EIGEN OFF)
if (${VECTORIZATION} STREQUAL "Enoki")
    set(USE_ENOKI ON)
    find_package(Enoki REQUIRED)
    include_directories(${ENOKI_INCLUDE_DIRS})
    message(STATUS "Enoki inlcude dirs : ${ENOKI_INCLUDE_DIRS}")
elseif(${VECTORIZATION} STREQUAL "Eigen")
    set(USE_EIGEN ON)
    find_package(Eigen3 REQUIRED)
    include_directories(${EIGEN3_INCLUDE_DIRS})
    message(STATUS "Eigen include dirs : ${EIGEN3_INCLUDE_DIRS}")
endif()

option(USE_PCG "Whether to use pcg random numbers" ON)
option(BUILD_TESTS "Whether to build tests" ON)
option(BUILD_PYTHON_BINDING "Whether to build python binding" ON)

include_directories(${CMAKE_SOURCE_DIR}/src)

if(WIN32)
    # Avoid compile error caused by min/max on WINDOWS
    add_definitions(-DNOMINMAX)
endif(WIN32)

find_package(OIIO REQUIRED)
include_directories(${OIIO_INCLUDE_DIRS})

find_package(OCIO REQUIRED)
include_directories(${OCIO_INCLUDE_DIRS})

find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIRS})

# TODO : Fix this when the vfxreferenceplatform has switched to oneapi
set(USE_ONEAPI OFF)
if(ONEAPI)
    set(USE_ONEAPI ON)
endif()

find_package(ASSIMP REQUIRED)
include_directories(${ASSIMP_INCLUDE_DIRS})

find_package(Boost 1.72 REQUIRED COMPONENTS filesystem)
include_directories(${Boost_INCLUDE_DIRS})

find_package(OpenEXR REQUIRED)
include_directories(${ILMBASE_INCLUDES})
include_directories(${OPENEXR_INCLUDES})

find_package(OSL REQUIRED)
include_directories(${OSL_INCLUDE_DIRS})

find_package(Frozen REQUIRED)
include_directories(${Frozen_INCLUDE_DIRS})

find_package(Pugixml REQUIRED)
include_directories(${Pugixml_INCLUDE_DIRS})

find_package(FMT REQUIRED)
include_directories(${FMT_INCLUDE_DIRS})

find_package(embree 3.13.0 REQUIRED)
include_directories(${EMBREE_INCLUDE_DIRS})

find_package(PCG32 REQUIRED)
include_directories(${PCG32_INCLUDE_DIRS})

include_directories("./src")

configure_file(resource/shader/include/kp_extension.h.in ${CMAKE_SOURCE_DIR}/resource/shader/include/kp_extension.h @ONLY)
configure_file(src/base/config.h.in include/config.h @ONLY)
include_directories(${CMAKE_BINARY_DIR}/include)

# OptiX related
option(USE_OPTIX "Whether to use optix" ON)
set(OPTIX_ROOT "" CACHE PATH "Installation path of OptiX SDK")
set(CUDA_TARGET_ARCH "sm_60" CACHE STRING "CUDA GPU architecture (e.g. sm_50)")

if(${USE_OPTIX})
    find_package(CUDA 8.0 REQUIRED)
    find_package(OptiX 7 REQUIRED)
    message(STATUS "Optix Include : ${OPTIX_INCLUDES}")
    include_directories(
        ${OPTIX_INCLUDES}
        ${CUDA_INCLUDE_DIRS})

    set(CUDA_INCLUDES ${CUDA_TOOLKIT_ROOT_DIR}/include)
    message(STATUS "CUDA include : ${CUDA_INCLUDES}")

    include(cuda_macros)
endif()
# End of OptiX related

set(BASE_HEADERS
    src/base/basic_types.h
    src/base/dictlike.h
    src/base/dpdf.h
    src/base/mat.h
    src/base/types.h
    src/base/vec.h
    src/base/utils.h)

set(CORE_HEADERS
    src/core/accel.h
    src/core/camera.h
    src/core/color.h
    src/core/film.h
    src/core/integrator.h
    src/core/light.h
    src/core/light_path.h
    src/core/material.h
    src/core/ray.h
    src/core/renderer.h
    src/core/sampler.h
    src/core/scene.h
    src/core/shape.h
    src/core/spectrum.h
    src/core/transform.h
    src/shading/bsdf.h
    src/shading/bsdfs.h
    src/shading/bssrdfs.h
    src/shading/compiler.h
    src/shading/querier.h
    src/shading/renderservices.h
    src/shading/shader.h
    src/shading/white_furnace.h
    src/shading/microfacet.h)

set(CORE_SOURCES
    src/core/accel.cpp
    src/core/camera.cpp
    src/core/film.cpp
    src/core/integrator.cpp
    src/core/light.cpp
    src/core/light_path.cpp
    src/core/material.cpp
    src/core/ray.cpp
    src/core/renderer.cpp
    src/core/sampler.cpp
    src/core/scene.cpp
    src/core/shape.cpp
    src/shading/bsdf.cpp
    src/shading/bsdfs.cpp
    src/shading/bssrdfs.cpp
    src/shading/compiler.cpp
    src/shading/querier.cpp
    src/shading/renderservices.cpp
    src/shading/shader.cpp
    src/shading/white_furnace.cpp
    src/shading/microfacet.cpp)

add_library(kazen_proto_lib STATIC
    ${CORE_SOURCES})
target_link_libraries(kazen_proto_lib
    ${ASSIMP_LIBRARIES}
    ${Boost_LIBRARIES}
    ${TBB_LIBRARIES}
    ${Pugixml_LIBRARIES}
    ${OCIO_LIBRARIES}
    ${OSL_LIBRARIES}
    ${EMBREE_LIBRARIES})

add_executable(kazen src/proto/main.cpp)
target_link_libraries(kazen
    ${OIIO_LIBRARIES}
    ${TBB_LIBRARIES}
    kazen_proto_lib)

if (BUILD_TESTS)
    add_subdirectory(tests)
endif()

if (BUILD_PYTHON_BINDING)
    add_subdirectory(src/binding)
endif()

set(OSL_HEADERS
    resource/shader/include/stdosl.h
    resource/shader/include/kp_extension.h)

file(GLOB OSL_SOURCES ${CMAKE_SOURCE_DIR}/resource/shader/src/*)

# Custom target to trigger all the osl compiling related commands
add_custom_target(
    compile_osl_shaders
    VERBATIM
)

# Echo a message
add_custom_command(TARGET compile_osl_shaders PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Compiling OSL shaders..."
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shader)

# Compile osl shaders into install directory
foreach(osl_source ${OSL_SOURCES})
    get_filename_component(osl_source_we ${osl_source} NAME_WE)
    set(oso_filename ${osl_source_we}.oso)
    set(OSLC_OPTIONS -O0 -DDEBUG)
    add_custom_command(
        TARGET compile_osl_shaders POST_BUILD
        COMMAND ${OSL_COMPILER} ${OSLC_OPTIONS}
            -I${CMAKE_SOURCE_DIR}/resource/shader/include
            -o ${CMAKE_BINARY_DIR}/shader/${oso_filename}
            ${osl_source}
        MAIN_DEPENDECY ${osl_source}
        DEPENDS ${OSL_HEADERS} ${OSL_SOURCES})
endforeach()

add_custom_target(
    shaders ALL
    DEPENDS ${OSL_HEADERS}
    SOURCES ${OSL_HEADERS} ${OSL_SOURCES}
    VERBATIM
)

add_dependencies(shaders compile_osl_shaders)

# Copy oslc into install dir
add_custom_command(
    TARGET compile_osl_shaders POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_INSTALL_PREFIX}/bin
    COMMAND ${CMAKE_COMMAND} -E copy ${OSL_COMPILER} ${CMAKE_INSTALL_PREFIX}/bin/)

# Installs
install(FILES ${OSL_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/shader/include)
install(DIRECTORY ${CMAKE_BINARY_DIR}/shader DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY ${CMAKE_SOURCE_DIR}/resource/texture DESTINATION ${CMAKE_INSTALL_PREFIX})
install(TARGETS kazen DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
install(DIRECTORY resource/scene/ DESTINATION ${CMAKE_INSTALL_PREFIX}/scene)