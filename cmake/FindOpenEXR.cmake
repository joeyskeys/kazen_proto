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
        /usr
        /usr/local
        ${OPENEXR_ROOT}
        $ENV{OPENEXR_ROOT}
    PATH_SUFFIXES
        lib)
    
find_library(OPENEXR_ILMTHREAD_LIBRARY IlmThread
    HINTS
        /usr
        /usr/local
        ${OPENEXR_ROOT}
        $ENV{OPENEXR_ROOT}
    PATH_SUFFIXES
        lib)

find_library(OPENEXR_ILMIMF_LIBRARY IlmImf
    HINTS
        /usr
        /usr/local
        ${OPENEXR_ROOT}
        $ENV{OPENEXR_ROOT}
    PATH_SUFFIXES
        lib)

find_library(OPENEXR_ILMIMFUTIL_LIBRARY IlmImfUtil
    HINTS
        /usr
        /usr/local
        ${OPENEXR_ROOT}
        $ENV{OPENEXR_ROOT}
    PATH_SUFFIXES
        lib)

find_library(OPENEXR_IEX_LIBRARY Iex
    HINTS
        /usr
        /usr/local
        ${OPENEXR_ROOT}
        $ENV{OPENEXR_ROOT}
    PATH_SUFFIXES
        lib)

find_library(OPENEXR_IEXMATH_LIBRARY IexMath
    HINTS
        /usr
        /usr/local
        ${OPENEXR_ROOT}
        $ENV{OPENEXR_ROOT}
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