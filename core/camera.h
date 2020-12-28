#pragma once

#include "base/vec.h"
#include "base/utils.h"

#include "film.h"
#include "ray.h"

class Camera {
public:
    Camera(const Vec3f& p,
        const Vec3f& l,
        const Vec3f& u,
        const float near,
        const float far,
        const float fov,
        const float ratio,
        Film* const film)
        : position(p)
        , lookat(l)
        , up(u)
        , near(near)
        , fov(far)
        , ratio(ratio)
        , film(film)
    {
        auto dir = (lookat - position).normalized();
        horizontal = dir.cross(up);
        vertical = horizontal.cross(dir);

        auto fov_in_radian = to_radian(fov);
        film_plane_height = near * std::tan(fov_in_radian);
        film_plane_width = film_plane_width * ratio;
        lower_left_corner = position
            + dir * near
            - film_plane_width * horizontal
            - film_plane_height * vertical;
    }

    Ray generate_ray(uint x, uint y) {
        // Default to perspective camera for now
        // fov denotes the vertical fov
        auto fov_in_radian = to_radian(fov);

        auto horizontal_vec = film_plane_width * (static_cast<float>(x) / film->width - 0.5f);
        auto vertical_vec = film_plane_height * (static_cast<float>(y) / film->height - 0.5f);

        auto direction = lower_left_corner
            + horizontal * film_plane_width * (static_cast<float>(x) / film->width)
            + vertical * film_plane_height * (static_cast<float>(y) / film->height);

        return Ray(position, direction.normalized());
    }

private:
    Vec3f position;
    Vec3f lookat;
    Vec3f up;
    Vec3f horizontal;
    Vec3f vertical;
    Vec3f lower_left_corner;
    float near;
    float far;
    float fov;
    float ratio;
    float film_plane_width;
    float film_plane_height;
    Film *film;
};