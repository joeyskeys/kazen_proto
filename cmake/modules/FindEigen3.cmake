find_path(EIGEN3_INCLUDE_DIR Eigen/Core
    HINTS
        /usr
        /usr/local
        ${EIGEN3_ROOT}
        $ENV{EIGEN3_ROOT}
    PATH_SUFFIXES
        include/eigen3)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Eigen3 DEFAULT_MSG
    EIGEN3_INCLUDE_DIR)

if(EIGEN3_FOUND)
    set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
else()
    set(EIGEN3_INCLUDE_DIRS)
    message(WARNING "Eigen3 not found")
endif()