#pragma once

#include <frozen/unordered_map.h>

#include "base/dictlike.h"
//#include "base/dual.h"
#include "base/mat.h"
#include "base/vec.h"
#include "base/utils.h"

#include "core/film.h"
#include "core/ray.h"

using base::Vec3f;
using base::Vec4f;
using base::Mat4f;

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
        , up(normalize(u))
        , near_plane(near_plane)
        , far_plane(near_plane)
        , fov(fov)
        , film(film)
    {
        init();
    }

    inline void init() {
        /*
        ratio = static_cast<float>(film->width) / static_cast<float>(film->height);

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
        center = position + dir * near_plane;
        */
        
        auto aspect = static_cast<float>(film->width) / static_cast<float>(film->height);
        auto z_axis = base::normalize(position - lookat);
        auto x_axis = base::normalize(base::cross(up, z_axis));
        auto y_axis = base::cross(z_axis, x_axis);
        camera_to_world = Mat4f{
            base::concat(x_axis, 0.f),
            base::concat(y_axis, 0.f),
            base::concat(z_axis, 0.f),
            base::concat(position, 1.f)
        };

        float recip = 1.f / (far_plane - near_plane);
        float cot = 1.f / std::tan(to_radian(fov / 2.f));
        auto proj_matrix = Mat4f{
            Vec4f{cot, 0, 0, 0},
            Vec4f{0, cot, 0, 0},
            Vec4f{0, 0, far_plane * recip, -near_plane * far_plane * recip},
            Vec4f{0, 0, 1, 0}
        };

        sample_to_camera = base::inverse(proj_matrix) * base::translate3f(Vec3f(-1.f, 1.f / aspect, 0.f)) *
            base::scale3f(Vec3f(2.f, -2.f / aspect, 1.f));
    }

    //Ray generate_ray(uint x, uint y);
    Ray generate_ray(const Vec2f pixel_sample) const;

    void scale(const Vec3f& s) {
        //horizontal *= s[0];
        //vertical *= s[1];
        camera_to_world *= base::scale3f(s);
    }

    void* address_of(const std::string& name) override;
    //void* runtime_address_of(const std::string& name);

public:
    Vec3f position;
    Vec3f lookat;
    Vec3f up;
    Vec3f horizontal;
    Vec3f vertical;
    Vec3f upper_left_corner;
    Vec3f center;
    float near_plane;
    float far_plane;
    float fov;
    float ratio;
    float film_plane_width;
    float film_plane_height;
    Film *film;

    Mat4f camera_to_world;
    Mat4f sample_to_camera;
};