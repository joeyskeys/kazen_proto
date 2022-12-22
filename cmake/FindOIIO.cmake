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

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OIIO DEFAULT_MSG
    OIIO_INCLUDE_DIRS
    OIIO_LIB)

if(OIIO_FOUND)
    set(OIIO_LIBRARIES ${OIIO_LIB})
endif()