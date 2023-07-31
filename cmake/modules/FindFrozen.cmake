find_path(Frozen_INCLUDE_DIRS frozen/map.h
    HINTS
        /usr
        /usr/local
        ${FROZEN_ROOT}
        $ENV{FROZEN_ROOT}
    PATH_SUFFIXES
        include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Frozen DEFAULT_MSG
    Frozen_INCLUDE_DIRS)
