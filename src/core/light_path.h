#pragma once

#include <algorithm>
#include <limits>
#include <ostream>

#include <OSL/oslexec.h>
#include <tbb/concurrent_vector.h>

#include "base/basic_types.h"
#include "base/dictlike.h"
#include "base/vec.h"
#include "core/intersection.h"
#include "core/spectrum.h"

class RecordContext {
public:
    uint pixel_x;
    uint pixel_y;
    uint depth;
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
    void record(const EventType t, const Intersection& isect, const RGBSpectrum& b, const RGBSpectrum& l);
    void output(std::ostream& os) const;
};

using LightPathPack = std::vector<LightPath>;
using RecorderDatabase = tbb::concurrent_vector<LightPathPack>;

class Recorder : public DictLike {
public:
    uint x_resolution;
    uint y_resolution;
    uint x_min = 0;
    uint x_max = std::numeric_limits<uint>::max();
    uint y_min = 0;
    uint y_max = std::numeric_limits<uint>::max();
    uint depth_min = 0;
    uint depth_max = std::numeric_limits<uint>::max();
    uint total_size;

    RecorderDatabase database;

    Recorder(uint w, uint h)
        : x_resolution(w)
        , y_resolution(h)
    {}

    void setup() {
        total_size = (x_max - x_min) *
            (y_max - y_min);
        database.resize(total_size);
    }

    void record(const LightPath& p, const RecordContext* ctx);
    void output(std::ostream& os) const;
    void print(const RecordContext* ctx, const std::string& str) const;

    void* address_of(const std::string& name) override;
};