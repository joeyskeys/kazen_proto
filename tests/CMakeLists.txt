
message("Test out")
add_executable(film_test film_test.cpp)
target_link_libraries(film_test
    ${OIIO_LIBRARIES}
    kazen_proto_lib)

message("oiio libs : ${OIIO_LIBRARIES} dir : ${OIIO_LIBRARY_DIR}")