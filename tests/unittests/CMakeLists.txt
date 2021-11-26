
find_package(Catch2 REQUIRED)

#add_executable(dictlike_tests dictlike_tests.cpp)
#target_link_libraries(dictlike_tests PRIVATE Catch2::Catch2WithMain)

add_executable(scene_parse_tests scene_parse_tests.cpp)
target_link_libraries(scene_parse_tests
    PRIVATE
        Catch2::Catch2WithMain
        kazen_proto_lib
        ${OIIO_LIBRARIES}
        ${TBB_LIBRARIES})