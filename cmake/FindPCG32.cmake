find_path(PCG32_INCLUDE_DIR pcg32.h
    HINTS
        /usr
        /usr/local
        ${PCG32_ROOT}
        $ENV{PCG32_ROOT}
    PATH_SUFFIXES
        include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PCG32 DEFAUT_MSG
    PCG32_INCLUDE_DIR)

if(PCG32_FOUND)
    set(PCG32_INCLUDE_DIRS ${PCG32_INCLUDE_DIR})
endif()