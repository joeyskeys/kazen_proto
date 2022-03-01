

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

void LightPath::record(const EventType t, const OSL::ShaderGlobals& sg,
    const RGBSpectrum& b, const RGBSpectrum& l) {
    LightPathEvent e;
    e.type = t;
    e.event_position = sg.P;
    e.ray_direction = sg.I;
    e.throughput = b;
    e.Li = l;
    path.emplace_back(std::move(e));
}

void Recorder::record(LightPath&& p, const RecordContext& ctx) {
    auto pack = pack_at(ctx);
    pack.emplace_back(std::move(p));
}

void Recorder::output(std::ostream& os) const {
    
}