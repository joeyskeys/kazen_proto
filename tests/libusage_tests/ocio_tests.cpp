#include <catch2/catch_all.hpp>

#include <OpenColorIO/OpenColorIO.h>

namespace OCIO = OCIO_NAMESPACE;
using Catch::Approx;

TEST_CASE("Color space transform", "transform") {
    auto config = OCIO::GetCurrentConfig();
    auto processor = config->getProcessor(OCIO::ROLE_MATTE_PAINT,
        OCIO::ROLE_SCENE_LINEAR);
    auto cpu_processor = processor->getDefaultCPUProcessor();

    float rgb[3] = { 0.39053256, 0.4777603, 0.64934449 };
    cpu_processor->applyRGB(rgb);
    REQUIRE(rgb[0] == Approx(0.16129f).epsilon(1e-3));
    REQUIRE(rgb[1] == Approx(0.19182530109362525f).epsilon(1e-3));
    REQUIRE(rgb[2] == Approx(0.3542388221f).epsilon(1e-3));
}