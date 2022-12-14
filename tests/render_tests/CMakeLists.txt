#add_executable(dictlike_tests dictlike_tests.cpp)
#target_link_libraries(dictlike_tests PRIVATE Catch2::Catch2WithMain)

add_executable(scene_parse_tests scene_parse_tests.cpp)
target_link_libraries(scene_parse_tests
    PRIVATE
        Catch2::Catch2WithMain
        kazen_proto_lib
        ${OIIO_LIBRARIES}
        ${TBB_LIBRARIES})

add_executable(integrator_tests integrator_tests.cpp)
target_link_libraries(integrator_tests
    PRIVATE
        kazen_proto_lib
        ${OIIO_LIBRARIES})

add_executable(microfacet_tests microfacet_tests.cpp)
target_link_libraries(microfacet_tests
    PRIVATE
        Catch2::Catch2WithMain
        kazen_proto_lib
        ${OIIO_LIBRARIES})

install(TARGETS scene_parse_tests integrator_tests
    DESTINATION ${CMAKE_INSTALL_PREFIX}/tests)