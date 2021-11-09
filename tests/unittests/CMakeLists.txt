
find_package(Catch2 3 REQUIRED)

add_executable(dictlike_tests dictlike_tests.cpp)
target_link_libraries(dictlike_tests PRIVATE Catch2::Catch2WithMain)