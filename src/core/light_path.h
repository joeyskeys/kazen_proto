#pragma once

#include <algorithm>
#include <limits>
#include <ostream>

#include <OSL/oslexec.h>
#include <tbb/concurrent_vector.h>

#include "base/basic_types.h"
#include "base/vec.h"
#include "core/spectrum.h"

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
        if (pixel_x >= designated_x_min && pixel_x < designated_x_max &&
            pixel_y >= designated_y_min && pixel_y < designated_y_max &&
            depth >= designated_depth_min && depth < designated_depth_max)
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

    void record(LightPathEvent&& e);
    void record(const EventType t, const OSL::ShaderGlobals& sg, const RGBSpectrum& b, const RGBSpectrum& l);
    void output(std::ostream& os) const;
};

using LightPathPack = std::vector<LightPath>;
//using RecorderDatabase = std::vector<LightPathPack>;
using RecorderDatabase = tbb::concurrent_vector<LightPathPack>;

class Recorder {
public:
    uint x_resolution;
    uint y_resolution;
    uint total_size;

    RecorderDatabase database;

    Recorder(uint w, uint h)
        : x_resolution(w)
        , y_resolution(h)
    {}

    void setup(const RecordContext& ctx) {
        total_size = (ctx.designated_x_max - ctx.designated_x_min) *
            (ctx.designated_y_max - ctx.designated_y_min);
        database.resize(total_size);
    }

    inline auto& pack_at(const RecordContext& ctx) {
        auto idx = (ctx.pixel_y - ctx.designated_y_min) *
            (ctx.designated_x_max - ctx.designated_x_min) +
            (ctx.pixel_x - ctx.designated_x_min);
        //std::cout << "x : " << ctx.pixel_x << ", y : " << ctx.pixel_y << std::endl;
        //std::cout << "database size : " << database.size() << ", idx : " << idx << std::endl;
        assert(database.size() > idx);
        return database.at(idx);
    }

    void record(LightPath&& p, const RecordContext& ctx);
    void output(std::ostream& os) const;
};