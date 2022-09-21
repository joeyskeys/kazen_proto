
add_executable(variant_tests variant_tests.cpp)
target_link_libraries(variant_tests PRIVATE Catch2::Catch2WithMain)

add_executable(vec_tests vec_tests.cpp)
target_link_libraries(vec_tests PRIVATE Catch2::Catch2WithMain)

add_executable(mat_tests mat_tests.cpp)
target_link_libraries(mat_tests PRIVATE Catch2::Catch2WithMain)

add_executable(dual_tests dual_tests.cpp)
target_link_libraries(dual_tests PRIVATE Catch2::Catch2WithMain)

install(TARGETS vec_tests mat_tests dual_tests
    DESTINATION ${CMAKE_INSTALL_PREFIX}/tests)