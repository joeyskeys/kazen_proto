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
