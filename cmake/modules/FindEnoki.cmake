find_path(ENOKI_INCLUDE_DIR enoki/array.h
    HITNS
        /usr
        /usr/local
        ${ENOKI_ROOT}
        $ENV{ENOKI_ROOT}
    PATH_SUFFIXES
        include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Enoki DEFAULT_MSG
    ENOKI_INCLUDE_DIR)

if(ENOKI_FOUND)
    set(ENOKI_INCLUDE_DIRS ${ENOKI_INCLUDE_DIR})
else()
    set(ENOKI_INCLUDE_DIRS)
    message(WARNING "Enoki not found")
endif()
