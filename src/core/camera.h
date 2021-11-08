#pragma once

#include <frozen/unordered_map.h>

#include "base/dictlike.h"
#include "base/vec.h"
#include "base/utils.h"

#include "film.h"
#include "ray.h"

class Camera : public DictLike {
public:
    Camera(const Vec3f& p,
        const Vec3f& l,
        const Vec3f& u,
        const float near,
        const float far,
        const float fov,
        Film* const film)
        : position(p)
        , lookat(l)
        , up(u.normalized())
        , near(near)
        , far(far)
        , fov(fov)
        , film(film)
    {
        ratio =  static_cast<float>(film->width) / static_cast<float>(film->height);

        auto dir = normalize(lookat - position);
        horizontal = normalize(dir.cross(up));
        vertical = normalize(dir.cross(horizontal));

        auto fov_in_radian = to_radian(fov);
        film_plane_height = near * std::tan(fov_in_radian);
        film_plane_width = film_plane_height * ratio;
        upper_left_corner = position
            + dir * near
            - film_plane_width * 0.5f * horizontal
            - film_plane_height * 0.5f * vertical;
    }

    Ray generate_ray(uint x, uint y);

    void* address_of(const std::string& name) override;
    void* runtime_address_of(const std::string& name);

private:
    Vec3f position;
    Vec3f lookat;
    Vec3f up;
    Vec3f horizontal;
    Vec3f vertical;
    Vec3f upper_left_corner;
    float near;
    float far;
    float fov;
    float ratio;
    float film_plane_width;
    float film_plane_height;
    Film *film;
};