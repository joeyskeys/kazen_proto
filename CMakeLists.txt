cmake_minimum_required(VERSION 3.12)

project(kazen-prototype LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/thirdparty)

find_package(OIIO REQUIRED)
include_directories(${OIIO_INCLUDE_DIRS})

find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIRS})

find_package(ASSIMP REQUIRED)
include_directories(${ASSIMP_INCLUDE_DIRS})

set(BASE_HEADERS
    base/mat.h
    base/types.h
    base/vec.h)

set(CORE_HEADERS
    core/color.h
    core/film.h
    core/ray.h
    core/spectrum.h
    core/transform.h
    core/camera.h
    core/integrator.h
    core/scene.h
    core/shape.h
    core/accel.h)

set(CORE_SOURCES
    core/camera.cpp
    core/film.cpp
    core/ray.cpp
    core/shape.cpp
    core/integrator.cpp
    core/material.cpp
    core/accel.cpp)

add_library(kazen_proto_lib
    ${CORE_SOURCES})
target_link_libraries(kazen_proto_lib
    ${ASSIMP_LIBRARIES})

add_executable(kazen proto/main.cpp)
target_link_libraries(kazen
    ${OIIO_LIBRARIES}
    ${TBB_LIBRARIES}
    kazen_proto_lib)

add_subdirectory(tests)
