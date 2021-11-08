#include "camera.h"
#include "sampling.h"

Ray Camera::generate_ray(uint x, uint y) {
    // Default to perspective camera for now
    // fov denotes the vertical fov
    auto fov_in_radian = to_radian(fov);

    auto direction = upper_left_corner
        + horizontal * film_plane_width * ((static_cast<float>(x) + randomf()) / film->width)
        + vertical * film_plane_height * ((static_cast<float>(y) + randomf()) / film->height)
        - position;

    return Ray(position, direction.normalized());
}

void* Camera::address_of(const std::string& name) override {
    constexpr frozen::unordered_map<frozen::string, int, 2> offset = {
        {"position", 0},
        {"lookat", sizeof(Vec3f)},
        {"up", sizeof(Vec3f) * 2},
        {"near", sizeof(Vec3f) * 6},
        {"far", sizeof(Vec3f) * 6 + sizeof(float)},
        {"fov", sizeof(Vec3f) * 6 + sizeof(float) * 2}
    };
    auto it = offset.find(name);
    if (it == offset.end())
        return nullptr;
    else
        return this + it->second;
}

void* Camera::runtime_address_of(const std::string& name) {
    if (name == "position")
        return &position;
    else if (name == "lookat")
        return &lookat;
    else if (name == "up")
        return &up;
    else if (name == "near")
        return &near;
    else if (name == "far")
        return &far;
    else if (name == "fov")
        return &fov;
}