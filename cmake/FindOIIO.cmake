find_path(OIIO_INCLUDE_DIRS OpenImageIO/imageio.h
    HINTS
        ${OIIO_ROOT}
        $ENV{OIIO_ROOT}
        /usr
        /usr/local
    PATH_SUFFIXES
        include)

find_library(OIIO_LIB OpenImageIO
    HINTS
        ${OIIO_ROOT}
        $ENV{OIIO_ROOT}
        /usr
        /usr/local
    PATH_SUFFIXES
        lib)

find_library(OIIO_UTIL_LIB OpenImageIO_Util
    HINTS
        ${OIIO_ROOT}
        $ENV{OIIO_ROOT}
        /usr
        /usr/local
    PATH_SUFFIXES
        lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OIIO DEFAULT_MSG
    OIIO_INCLUDE_DIRS
    OIIO_LIB
    OIIO_UTIL_LIB)

if(OIIO_FOUND)
    set(OIIO_LIBRARIES ${OIIO_LIB} ${OIIO_UTIL_LIB})
    message(STATUS "OIIO libs : ${OIIO_LIBRARIES}")
endif()