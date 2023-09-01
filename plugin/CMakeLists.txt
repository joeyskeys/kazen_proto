add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/bazen.zip
    COMMAND ${CMAKE_COMMAND} -E tar cfv ${CMAKE_BINARY_DIR}/bazen.zip
        --format=zip ${CMAKE_CURRENT_SOURCE_DIR}/bazen
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bazen copylib)

add_custom_target(create_zip ALL
    DEPENDS ${CMAKE_BINARY_DIR}/bazen.zip)