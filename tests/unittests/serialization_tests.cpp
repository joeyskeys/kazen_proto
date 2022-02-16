#include <string>

#include <catch2/catch.hpp>
#include <pugixml.hpp>

#include "core/camera.h"

using namespace std::string_literals;

TEST_CASE("Member reflect", "[single-file") {
    Camera cam{};

    auto fov = hana::at_key(cam, BOOST_HANA_STRING("fov"));

    pugi::xml_document doc;
    pugi::xml_parse_result ret = doc.load_file("../tests/unittests/camera_attributes.xml");
    for (auto& node : doc) {
        auto member = hana::at_key(cam, BOOST_HANA_STRING(node.name()));
        std::cout << "member " << node.name() << " : " << member << std::endl;
    }

    REQUIRE(fov == 60);
}