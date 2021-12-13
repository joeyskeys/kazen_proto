#pragma once

#include <frozen/unordered_map.h>

#include "base/dictlike.h"
#include "base/vec.h"
#include "base/utils.h"

#include "film.h"
#include "ray.h"

class Camera : public DictLike {
public:
    Camera()
        : position(Vec3f(0, 0, 5))
        , lookat(Vec3f(0, 0, 0))
        , up(Vec3f(0, 1, 0))
        , near_plane(1)
        , far_plane(1000)
        , fov(60)
        , film(nullptr)
    {}

    Camera(const Vec3f& p,
        const Vec3f& l,
        const Vec3f& u,
        const float near_plane,
        const float far_plane,
        const float fov,
        Film* const film)
        : position(p)
        , lookat(l)
        , up(u.normalized())
        , near_plane(near_plane)
        , far_plane(near_plane)
        , fov(fov)
        , film(film)
    {
        init();
    }

    inline void init() {
        ratio =  static_cast<float>(film->width) / static_cast<float>(film->height);

        auto dir = normalize(lookat - position);
        horizontal = normalize(dir.cross(up));
        vertical = normalize(dir.cross(horizontal));

        auto fov_in_radian = to_radian(fov);
        film_plane_height = near_plane * std::tan(fov_in_radian);
        film_plane_width = film_plane_height * ratio;
        upper_left_corner = position
            + dir * near_plane
            - film_plane_width * 0.5f * horizontal
            - film_plane_height * 0.5f * vertical;
    }

    Ray generate_ray(uint x, uint y);

    void* address_of(const std::string& name) override;
    //void* runtime_address_of(const std::string& name);

public:
    Vec3f position;
    Vec3f lookat;
    Vec3f up;
    Vec3f horizontal;
    Vec3f vertical;
    Vec3f upper_left_corner;
    float near_plane;
    float far_plane;
    float fov;
    float ratio;
    float film_plane_width;
    float film_plane_height;
    Film *film;
};