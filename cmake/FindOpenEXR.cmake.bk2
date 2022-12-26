find_package(Imath CONFIG)
if (NOT TARGET Imath::Imath)
    find_package(Ilmbase CONFIG)
endif()
find_package(OpenEXR CONFIG)

find_path(OPENEXR_INCLUDE_DIR OpenEXR/OpenEXRConfig.h
    HINTS
        ${OPENEXR_ROOT}
        $ENV{OPENEXR_ROOT}
        /usr
        /usr/local
    PATH_SUFFIXES
        include)


find_library(OPENEXR_HALF_LIBRARY Half
    HINTS
        ${OPENEXR_ROOT}
        $ENV{OPENEXR_ROOT}
        /usr
        /usr/local
    PATH_SUFFIXES
        lib)
    
find_library(OPENEXR_ILMTHREAD_LIBRARY IlmThread
    HINTS
        ${OPENEXR_ROOT}
        $ENV{OPENEXR_ROOT}
        /usr
        /usr/local
    PATH_SUFFIXES
        lib)

find_library(OPENEXR_ILMIMF_LIBRARY IlmImf
    HINTS
        ${OPENEXR_ROOT}
        $ENV{OPENEXR_ROOT}
        /usr
        /usr/local
    PATH_SUFFIXES
        lib)

find_library(OPENEXR_ILMIMFUTIL_LIBRARY IlmImfUtil
    HINTS
        ${OPENEXR_ROOT}
        $ENV{OPENEXR_ROOT}
        /usr
        /usr/local
    PATH_SUFFIXES
        lib)

find_library(OPENEXR_IEX_LIBRARY Iex
    HINTS
        ${OPENEXR_ROOT}
        $ENV{OPENEXR_ROOT}
        /usr
        /usr/local
    PATH_SUFFIXES
        lib)

find_library(OPENEXR_IEXMATH_LIBRARY IexMath
    HINTS
        ${OPENEXR_ROOT}
        $ENV{OPENEXR_ROOT}
        /usr
        /usr/local
    PATH_SUFFIXES
        lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenEXR DEFAULT_MSG
    OPENEXR_INCLUDE_DIR
    OPENEXR_HALF_LIBRARY
    OPENEXR_ILMTHREAD_LIBRARY
    OPENEXR_ILMIMF_LIBRARY
    OPENEXR_ILMIMFUTIL_LIBRARY
    OPENEXR_IEX_LIBRARY
    OPENEXR_IEXMATH_LIBRARY)

if(OPENEXR_FOUND)
    set(OPENEXR_INCLUDES ${OPENEXR_INCLUDE_DIR})
    set(OPENEXR_LIBRARIES ${OPENEXR_HALF_LIBRARY} ${OPENEXR_IEX_LIBRARY}
        ${OPENEXR_IEXMATH_LIBRARY} ${OPENEXR_ILMTHREAD_LIBRARY} ${OPENEXR_ILMIMF_LIBRARY}
        ${OPENEXR_ILMIMFUTIL_LIBRARY})
else()
    set(OPENEXR_INCLUDES)
    set(OPENEXR_LIBRARIES)
    message(WARNING "OpenEXR not found")
endif()