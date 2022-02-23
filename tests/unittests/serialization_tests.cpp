#include <string>
#include <type_traits>

#include <catch2/catch.hpp>
#include <pugixml.hpp>

#include "core/camera.h"

using namespace std::string_literals;

TEST_CASE("Member reflect", "[single-file") {
    Camera cam{};

    // Compile time for loop will unfold the whole loop and create a huge binary..
    hana::for_each(hana::accessors<Camera>(), hana::fuse([&](auto name, auto accessor) {
        auto runtime_str = std::string(name.c_str());
        auto& tmp = accessor(cam);
        std::cout << "handling : " << runtime_str << std::endl;

        // Compile time comparison to eliminate redundent code for other members
        // or compilation will fail at the line "tmp = 70.f" coz tmp is STRONG TYPED
        //if constexpr (name == BOOST_HANA_STRING("fov")) {
        if constexpr (std::is_arithmetic_v<std::decay_t<decltype(tmp)>>) {

            // Runtime comparison to actually find the member we're interested
            if (runtime_str == "fov") {
                tmp = 70.f;
                std::cout << "we got luck here" << std::endl;
            }
        }
    }));

    /*
    pugi::xml_document doc;
    pugi::xml_parse_result ret = doc.load_file("../tests/unittests/camera_attributes.xml");
    for (auto& node : doc) {
        auto node_name = node.name();
        std::cout << "node name : " << node_name << std::endl;
        // Compile time computation only..
        auto member = hana::at_key(cam, BOOST_HANA_STRING(node_name));
        //std::cout << "member " << node.name() << " : " << member << std::endl;
    }
    */

    auto fov = hana::at_key(cam, BOOST_HANA_STRING("fov"));
    REQUIRE(fov == 60);
}