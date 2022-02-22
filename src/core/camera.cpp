#include "camera.h"
#include "sampling.h"

Ray Camera::generate_ray(uint x, uint y) {
    // Default to perspective camera for now
    // fov denotes the vertical fov
    auto fov_in_radian = to_radian(fov);

    auto direction = upper_left_corner
        + horizontal * film_plane_width * (static_cast<float>(x) + randomf()) / film->width
        + vertical * film_plane_height * (static_cast<float>(y) + randomf()) / film->height
        - position;
    /*
    auto direction = upper_left_corner
        + horizontal * film_plane_width * (Dual2f(static_cast<float>(x) + randomf(), 1, 0) / film->width)
        + vertical * film_plane_height * (Dual2f(static_cast<float>(y) + randomf(), 0, 1) / film->height)
        - position;
    */

    return Ray(position, direction.normalized());
    //return Ray(position, normalize(direction));
}

void* Camera::address_of(const std::string& name) {
    if (name == "position")
        return &position;
    else if (name == "lookat")
        return &lookat;
    else if (name == "up")
        return &up;
    else if (name == "near")
        return &near_plane;
    else if (name == "far")
        return &far_plane;
    else if (name == "fov")
        return &fov;
    return nullptr;
}