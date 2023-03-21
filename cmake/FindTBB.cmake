find_path(TBB_INCLUDE_DIRS tbb/tbb.h
    HINTS
        /usr
        /usr/local
    PATH_SUFFIXES
        include)

find_path(TBB_VERSION_PATH tbb/version.h
    HINTS
        /usr
        /usr/local
    PATH_SUFFIXES
        include)

set(ONEAPI ON)
if(TBB_VERSION_PATH)
    set(ONEAPI ON)
endif()

find_library(TBB_PROXY_LIB tbbmalloc_proxy
    HINTS
        /usr
        /usr/local
    PATH_SUFFIXES
        lib)

find_library(TBB_MALLOC_LIB tbbmalloc
    HINTS
        /usr
        /usr/local
    PATH_SUFFIXES
        lib)

find_library(TBB_LIB tbb
    HITNS
        /usr
        /usr/local
    PATH_SUFFIXES
        lib)


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TBB DEFAULT_MSG
    TBB_INCLUDE_DIRS
    TBB_PROXY_LIB
    TBB_MALLOC_LIB
    TBB_LIB)

if(TBB_FOUND)
    set(TBB_LIBRARIES ${TBB_PROXY_LIB} ${TBB_MALLOC_LIB} ${TBB_LIB})
endif()