
add_executable(ocio_tests ocio_tests.cpp)
target_link_libraries(ocio_tests
    PRIVATE
        Catch2::Catch2WithMain ${OCIO_LIBRARIES})

set(cuda_srcs
    solid.cu)

set(cuda_headers
    solid.h)

foreach (cuda_src ${cuda_srcs})
    NVCC_COMPILE(${cuda_src} "" ptx_generated "")
    list (APPEND ptx_list ${ptx_generated})
endforeach()

add_executable(embree_tests embree_tests.cpp)
target_link_libraries(embree_tests
    PRIVATE
        Catch2::Catch2WithMain
        ${EMBREE_LIBRARIES})

add_custom_target(libusage_ptx ALL
    DEPENDS ${ptx_list}
    SOURCES ${cuda_srcs})

add_executable(optix_tests optix_tests.cpp)
target_link_libraries(optix_tests
    PRIVATE
        Catch2::Catch2WithMain CUDA::cudart)