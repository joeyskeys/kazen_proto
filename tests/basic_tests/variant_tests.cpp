#include <variant>
#include <string>

#include <catch2/catch.hpp>

#include "base/vec.h"

using namespace base;
using namespace std::literals;

using Comb = std::variant<std::string, int, float, Vec3f>;

void fill_content_1(Comb* combptr) {
    *combptr = "test1"s;
}

void fill_content_2(void* ptr) {
    auto typed_ptr = reinterpret_cast<std::string*>(ptr);
    *typed_ptr = "test2sdfsdfsdfsdfsdfsdfsdfsdfsdfsdfsdfsdf"s;
}

TEST_CASE("Variant test", "[single-file]") {
    Comb comb;

    fill_content_1(&comb);
    REQUIRE(std::get<std::string>(comb) == "test1");

    fill_content_2(&comb);
    REQUIRE(std::get<std::string>(comb) == "test2sdfsdfsdfsdfsdfsdfsdfsdfsdfsdfsdfsdf");

    void* vptr = &comb;
    auto strptr = reinterpret_cast<std::string*>(vptr);
    REQUIRE(*strptr == "test2sdfsdfsdfsdfsdfsdfsdfsdfsdfsdfsdfsdf");
}