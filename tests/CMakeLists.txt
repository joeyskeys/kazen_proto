
add_executable(film_test film_test.cpp)
target_link_libraries(film_test
    ${OIIO_LIBRARIES}
    kazen_proto_lib)

add_executable(mat_test mat_test.cpp)
add_executable(vec_test vec_test.cpp)
