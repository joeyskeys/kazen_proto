
add_executable(ocio_tests ocio_tests.cpp)
target_link_libraries(ocio_tests
    PRIVATE
        Catch2::Catch2WithMain ${OCIO_LIBRARIES})

add_executable(optix_tests optix_tests.cpp)
target_link_libraries(optix_tests
    PRIVATE
        Catch2::Catch2WithMain ${CUDA_LIBRARIES})