add_executable(scene_api_tests scene_api_test.cpp)
target_link_libraries(scene_api_tests
    PRIVATE
        Catch2::Catch2WithMain
        kazen_proto_lib
        ${OIIO_LIBRARIES}
        ${TBB_LIBRARIES})