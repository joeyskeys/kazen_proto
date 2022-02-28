#pragma once

#include <ostream>

#include "base/basic_types.h"
#include "base/vec.h"
#include "core/spectrum.h"

#include <algorithm>
#include <limits>

#include <OSL/oslexec.h>

class RecordContext {
public:
    uint pixel_x;
    uint pixel_y;
    uint designated_x_min = 0;
    uint designated_x_max = std::numeric_limits<uint>::max();
    uint designated_y_min = 0;
    uint designated_y_max = std::numeric_limits<uint>::max();
    uint depth;
    uint designated_depth_min = 0;
    uint designated_depth_max = std::numeric_limits<uint>::max();

    inline bool should_record() const {
        if (pixel_x == std::clamp(pixel_x, designated_x_min, designated_x_max) &&
            pixel_y == std::clamp(pixel_y, designated_y_min, designated_y_max) &&
            depth == std::clamp(depth, designated_depth_min, designated_depth_max))
            return true;
        return false;
    }
};

enum EventType {
    // Special ones
    EStart,
    EMaxDepth,
    ERouletteCut,
    EBackground,
    // Hit
    EReflection,
    ETransmission,
    EVolume,
    EEmission
};

inline constexpr std::array<const char*, 8> event_strings {
    "Start",
    "MaxDepth",
    "RouletteCut",
    "Background",
    "Reflection",
    "Transmission",
    "Volume",
    "Emission"
};

class LightPathEvent {
public:
    // Type
    EventType       type;

    // Geometric information
    Vec3f           event_position;
    Vec3f           ray_origin;
    Vec3f           ray_direction;

    // Hit object information
    uint            hit_geom_id;
    Vec3f           normal;
    Vec2f           uv;

    // Light path information
    RGBSpectrum     throughput;
    RGBSpectrum     Li;

};

class LightPath {
public:
    std::vector<LightPathEvent> path;

    void output(std::ostream& os) const;
    void record(LightPathEvent&& e);
    void record(const EventType t, const OSL::ShaderGlobals& sg, const RGBSpectrum& b, const RGBSpectrum& l);
};

using LightPathPack = std::vector<LightPath>;
using RecorderDatabase = std::vector<LightPathPack>;

class Recorder {
public:
    uint x_resolution;
    uint y_resolution;

    RecordContext*   ctx;
    RecorderDatabase database;

    Recorder(uint w, uint h, RecordContext* c=nullptr)
        : x_resolution(w)
        , y_resolution(h)
        , ctx(c)
    {
        database.reserve(w * h);
    }

    inline auto& pack_at(uint x, uint y) {
        assert(database.size() > x_resolution * y_resolution);
        return database[y * y_resolution + x];
    }

    void record(LightPath&& p);
};