cmake_minimum_required(VERSION 3.12)

project(kazen-prototype LANGUAGES CXX C)

cmake_policy(SET CMP0074 NEW)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "set build type")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

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

configure_file(resource/shader/kp_extension.h.in shader/kp_extension.h @ONLY)
configure_file(src/base/config.h.in include/config.h @ONLY)
include_directories(${CMAKE_BINARY_DIR}/include)

include_directories(${CMAKE_SOURCE_DIR}/src)

if(WIN32)
    # Avoid compile error caused by min/max on WINDOWS
    add_definitions(-DNOMINMAX)
endif(WIN32)

find_package(OIIO REQUIRED)
include_directories(${OIIO_INCLUDE_DIRS})

find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIRS})

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

include_directories("./src")

set(BASE_HEADERS
    src/base/basic_types.h
    src/base/dictlike.h
    src/base/dpdf.h
    src/base/mat.h
    src/base/types.h
    src/base/vec.h)

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
    src/core/scene.h
    src/core/shape.h
    src/core/spectrum.h
    src/core/transform.h
    src/shading/bsdf.h
    src/shading/bsdfs.h
    src/shading/compiler.h
    src/shading/renderservices.h
    src/shading/shader.h
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
    src/core/scene.cpp
    src/core/shape.cpp
    src/shading/bsdf.cpp
    src/shading/bsdfs.cpp
    src/shading/compiler.cpp
    src/shading/renderservices.cpp
    src/shading/shader.cpp
    src/shading/microfacet.cpp)

add_library(kazen_proto_lib
    ${CORE_SOURCES})
target_link_libraries(kazen_proto_lib
    ${ASSIMP_LIBRARIES}
    ${Boost_LIBRARIES}
    ${TBB_LIBRARIES}
    ${Pugixml_LIBRARIES}
    ${OSL_LIBRARIES}
    ${EMBREE_LIBRARIES})

add_executable(kazen src/proto/main.cpp)
target_link_libraries(kazen
    ${OIIO_LIBRARIES}
    ${TBB_LIBRARIES}
    kazen_proto_lib)

add_subdirectory(tests)

set(OSL_HEADERS
    resource/shader/stdosl.h)

install(FILES ${OSL_HEADERS} DESTINATION ${CMAKE_BINARY_DIR}/shader)
