find_path(OCIO_INCLUDE_DIRS OpenColorIO/OpenColorIO.h
    HINTS
        /usr
        /usr/local
    PATH_SUFFIXES
        include)

find_library(OCIO_LIB OpenColorIO
    HINTS
        /usr
        /usr/local
    PATH_SUFFIXES
        lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OCIO DEFAULT_MSG
    OCIO_INCLUDE_DIRS
    OCIO_LIB)

if(OCIO_FOUND)
    set(OCIO_LIBRARIES ${OCIO_LIB})
endif()