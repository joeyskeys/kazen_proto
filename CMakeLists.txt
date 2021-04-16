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

find_package(Boost 1.72 REQUIRED COMPONENTS filesystem)
include_directories(${Boost_INCLUDE_DIRS})

message("Boost libs ${Boost_LIBRARY_DIRS}")

set(BASE_HEADERS
    base/mat.h
    base/types.h
    base/vec.h)

set(CORE_HEADERS
    core/accel.h
    core/camera.h
    core/color.h
    core/film.h
    core/integrator.h
    core/material.h
    core/ray.h
    core/scene.h
    core/shape.h
    core/spectrum.h
    core/transform.h)

set(CORE_SOURCES
    core/accel.cpp
    core/camera.cpp
    core/film.cpp
    core/integrator.cpp
    core/material.cpp
    core/ray.cpp
    core/shape.cpp)

add_library(kazen_proto_lib
    ${CORE_SOURCES})
target_link_libraries(kazen_proto_lib
    ${ASSIMP_LIBRARIES}
    ${Boost_LIBRARIES})

add_executable(kazen proto/main.cpp)
target_link_libraries(kazen
    ${OIIO_LIBRARIES}
    ${TBB_LIBRARIES}
    kazen_proto_lib)

add_subdirectory(tests)
