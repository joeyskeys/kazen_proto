if(${OPTIX_ROOT} STREQUAL "")
    message(ERROR "OPTIX_ROOT not set")
endif()

find_path(OPTIX_INCLUDE_DIR
    NAMES optix.h
    HINTS
        ${OPTIX_ROOT}
        $ENV{OPTIX_ROOT}
    PATH_SUFFIXES
        include)

find_path(OPTIX_SDK_DIR
    NAMES cuda/helpers.h
    HINTS
        ${OPTIX_ROOT}/SDK
        $ENV{OPTIX_ROOT}/SDK)

# We default to Optix 7 which DONT have any libraries

if(OPTIX_INCLUDE_DIR)
    # Pull out the API version from optix.h
    file(STRINGS ${OPTIX_INCLUDE_DIR}/optix.h OPTIX_VERSION_LINE LIMIT_COUNT 1 REGEX "define OPTIX_VERSION")
    string(REGEX MATCH "([0-9]+)" OPTIX_VERSION_STRING "${OPTIX_VERSION_LINE}")
    math(EXPR OPTIX_VERSION_MAJOR "${OPTIX_VERSION_STRING}/10000")
    math(EXPR OPTIX_VERSION_MINOR "(${OPTIX_VERSION_STRING}%10000)/100")
    math(EXPR OPTIX_VERSION_MICRO "${OPTIX_VERSION_STRING}%100")
    set(OPTIX_VERSION "${OPTIX_VERSION_MAJOR}.${OPTIX_VERSION_MINOR}.${OPTIX_VERSION_MICRO}")
endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OptiX
    FOUND_VAR OPTIX_FOUND
    REQUIRED_VARS OPTIX_INCLUDE_DIR OPTIX_VERSION OPTIX_SDK_DIR
    VERSION_VAR OPTIX_VERSION)

if(OPTIX_FOUND)
    set(OPTIX_INCLUDES ${OPTIX_INCLUDE_DIR})
    set(OPTIX_SDK_INCLUDES ${OPTIX_SDK_DIR})
endif()