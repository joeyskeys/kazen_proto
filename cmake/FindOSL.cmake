find_path(OSL_INCLUDE_DIR OSL/oslexec.h
    HINTS
        /usr
        /usr/local
    PATH_SUFFIXES
        include)

find_library(OSL_EXEC_LIBRARY oslexec
    HINTS
        /usr
        /usr/local
    PATH_SUFFIXES
        lib)


find_library(OSL_COMP_LIBRARY oslcomp
    HINTS
        /usr
        /usr/local
    PATH_SUFFIXES
        lib)

find_library(OSL_QUERY_LIBRARY oslquery
    HINTS
        /usr
        /usr/local
    PATH_SUFFIXES
        lib)

find_program(OSL_COMPILER oslc
    HINTS
        /usr
        /usr/local
    PATH_SUFFIXES
        bin)

find_program(OSL_QUERY_INFO oslinfo
    HINTS
        /usr
        /usr/local
    PATH_SUFFIXES
        bin)

find_program(OSL_MAKETX maketx
    HINTS
        /usr
        /usr/local
    PATH_SUFFIXES
        bin)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OSL DEFAULT_MSG
    OSL_INCLUDE_DIR
    OSL_EXEC_LIBRARY
    OSL_COMP_LIBRARY
    OSL_QUERY_LIBRARY
    OSL_COMPILER
    OSL_QUERY_INFO
    OSL_MAKETX)

if(OSL_FOUND)
    set(OSL_INCLUDE_DIRS ${OSL_INCLUDE_DIR})
    set(OSL_LIBRARIES ${OSL_EXEC_LIBRARY} ${OSL_COMP_LIBRARY} ${OSL_QUERY_LIBRARY})
else()
    set(OSL_INCLUDE_DIRS)
    set(OSL_LIBRARIES)
endif()
