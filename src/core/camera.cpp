#include "core/camera.h"
#include "core/sampling.h"

Ray Camera::generate_ray(uint x, uint y) {
    // Default to perspective camera for now
    // fov denotes the vertical fov
    /*
    auto fov_in_radian = to_radian(fov);

    auto direction = center
        + horizontal * film_plane_width * (static_cast<float>(x) + randomf() - film->width / 2) / film->width
        + vertical * film_plane_height * (static_cast<float>(y) + randomf() - film->height / 2) / film->height
        - position;

    return Ray(position, direction.normalized());
    */

    Vec3f near_p = (sample_to_camera * Vec4f(
        (static_cast<float>(x) + randomf()) / film->width,
        (static_cast<float>(y) + randomf()) / film->height,
        0.f, 1.f)).reduct<3>();
    auto d = Vec4f{near_p.normalized(), 0.f};
    float inv_z = 1.f / d.z();

    return Ray((camera_to_world * Vec4f{0.f, 0.f, 0.f, 1.f}).reduct<3>(), (camera_to_world * d).reduct<3>());
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