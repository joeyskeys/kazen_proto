#include <catch2/catch.hpp>

#include <OpenColorIO/OpenColorIO.h>

namespace OCIO = OCIO_NAMESPACE;

TEST_CASE("Color space transform", "transform") {
    auto config = OCIO::GetCurrentConfig();
    auto processor = config->getProcessor(OCIO::ROLE_COMPOSITING_LOG,
        OCIO::ROLE_SCENE_LINEAR);
    auto cpu_processor = processor->getDefaultCPUProcessor();

    float rgb[3] = { 0.39053256, 0.4777603, 0.64934449 };
    cpu_processor->applyRGB(rgb);
    REQUIRE(rgb[0] == Approx(16.1010133f));
}