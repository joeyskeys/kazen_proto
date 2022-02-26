#pragma once

#include <ostream>

#include "base/basic_types.h"
#include "base/vec.h"
#include "core/spectrum.h"

class RecContext {
public:
    uint pixel_x;
    uint pixel_y;
    uint depth;

    bool should_record();
};

enum EventType {
    // Special ones
    Start,
    MaxDepth,
    RouletteCut,
    Background,
    // Hit
    Reflection,
    Transmission,
    Volume,
    Emission
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

    void record(std::ostream& o);
};