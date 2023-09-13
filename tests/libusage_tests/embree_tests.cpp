#include <array>
#include <limits>
#include <stdint.h>

#include <catch2/catch_all.hpp>

#include <embree4/rtcore.h>

TEST_CASE("Embree intersection", "intersection") {
    auto device = rtcNewDevice(nullptr);
    auto scene = rtcNewScene(device);
    rtcSetSceneFlags(scene, RTC_SCENE_FLAG_ROBUST);
    rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_HIGH);

    auto subscene = rtcNewScene(device);
    rtcSetSceneFlags(scene, RTC_SCENE_FLAG_ROBUST);
    rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_HIGH);

    std::array<float, 9> verts = {
        -1, 0, 1,
         2, 0, 1,
        -1, 0, -2
    };
    std::array<uint32_t, 3> indice = {
        0, 1, 2
    };
    std::array<float, 16> matrix = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };

    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
    rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
        verts.data(), 0, sizeof(float) * 3, 3);
    rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
        indice.data(), 0, sizeof(uint32_t) * 3, 1);

    rtcCommitGeometry(geom);
    auto geom_id = rtcAttachGeometry(subscene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(subscene);

    RTCGeometry instance = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE);
    rtcSetGeometryInstancedScene(instance, subscene);
    rtcSetGeometryTimeStepCount(instance, 1);
    rtcSetGeometryTransform(instance, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR,
        matrix.data());
    rtcCommitGeometry(instance);
    auto ins_id = rtcAttachGeometry(scene, instance);
    rtcReleaseGeometry(instance);
    rtcReleaseScene(subscene);

    rtcCommitScene(scene);

    RTCIntersectArguments iargs;
    rtcInitIntersectArguments(&iargs);

    RTCRayHit rayhit;
    rayhit.ray.org_x = 0;
    rayhit.ray.org_y = 1;
    rayhit.ray.org_z = 0;
    rayhit.ray.dir_x = 0;
    rayhit.ray.dir_y = -1;
    rayhit.ray.dir_z = 0;
    rayhit.ray.tnear = 0;
    rayhit.ray.tfar = std::numeric_limits<float>::max();
    rayhit.ray.mask = -1;
    rayhit.ray.flags = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

    rtcIntersect1(scene, &rayhit, &iargs);
    REQUIRE(rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID);
    REQUIRE(rayhit.hit.geomID == geom_id);
}