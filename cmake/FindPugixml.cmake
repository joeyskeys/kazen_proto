find_path(Pugixml_INCLUDE_DIRS pugixml.hpp
    HINTS
        /usr
        /usr/local
    PATH_SUFFIXES
        include)

find_library(Pugixml_LIB pugixml
    HINTS
        /usr
        /usr/local
    PATH_SUFFIXES
        /lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Pugixml DEFAULT_MSG
    Pugixml_INCLUDE_DIRS
    Pugixml_LIB)

if(Pugixml_FOUND)
    set(Pugixml_LIBRARIES ${Pugixml_LIB})
endif()