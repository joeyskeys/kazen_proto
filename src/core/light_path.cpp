

#include "light_path.h"

void LightPath::output(std::ostream& os) const {
    // Output in the multithread context will cause problem
    for (const auto& e : path) {
        os << "Event Type : " << event_strings[e.type]
            << ", Event Position : " << e.event_position
            << ", Next Direction : " << e.ray_direction
            << ", throughput : " << e.throughput
            << ", Li : " << e.Li << std::endl;
    }
}

void LightPath::record(LightPathEvent&& e) {
    path.emplace_back(std::move(e));
}

void LightPath::record(const EventType t, const Intersection& isect,
    const RGBSpectrum& b, const RGBSpectrum& l) {
    LightPathEvent e;
    e.type = t;
    e.event_position = isect.refined_point;
    e.ray_direction = isect.wi;
    e.hit_geom_id = isect.geom_id;
    e.throughput = b;
    e.Li = l;
    path.emplace_back(e);
}

void Recorder::record(const LightPath& p, const RecordContext* ctx) {
    if (ctx && (ctx->pixel_x < x_min || ctx->pixel_x >= x_max ||
        ctx->pixel_y < y_min || ctx->pixel_y >= y_max ||
        ctx->depth < depth_min || ctx->depth >= depth_max))
        return;

    auto idx = (ctx->pixel_y - y_min) * (x_max - x_min) + (ctx->pixel_x - x_min);
    assert(database.size() > idx);
    database.at(idx).emplace_back(p);
}

void Recorder::output(std::ostream& os) const {
    std::cout << "database size : " << database.size() << std::endl;
    for (const auto& pack : database) {
        std::cout << "pack size : " << pack.size() << std::endl;
        for (const auto& pathobj : pack) {
            std::cout << "path element size : " << pathobj.path.size() << std::endl;
            for (const auto& e : pathobj.path) {
                os << "Event Type : " << event_strings[e.type]
                    << ", Event Position : " << e.event_position
                    << ", Next Direction : " << e.ray_direction
                    << ", Geom ID : " << e.hit_geom_id
                    << ", throughput : " << e.throughput
                    << ", Li : " << e.Li << std::endl;
            }

            os << std::endl;
        }
    }
}

void Recorder::print(const RecordContext* ctx, const std::string& str) const {
    if (ctx->pixel_x < x_min || ctx->pixel_x >= x_max ||
        ctx->pixel_y < y_min || ctx->pixel_y >= y_max ||
        ctx->depth < depth_min || ctx->depth >= depth_max)
        return;
        
    std::cout << str << std::endl;
}

void* Recorder::address_of(const std::string& name) {
    if (name == "x_resolution")
        return &x_resolution;
    else if (name == "y_resolution")
        return &y_resolution;
    else if (name == "x_min")
        return &x_min;
    else if (name == "y_max")
        return &x_max;
    else if (name == "y_min")
        return &y_min;
    else if (name == "y_max")
        return &y_max;
    return nullptr;
}