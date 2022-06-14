#pragma once

#include <frozen/unordered_map.h>

#include "base/dictlike.h"
//#include "base/dual.h"
#include "base/mat.h"
#include "base/vec.h"
#include "base/utils.h"

#include "core/film.h"
#include "core/ray.h"

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
        /*
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
        center = position + dir * near_plane;
        */
        auto z_axis = (position - lookat).normalized();
        auto x_axis = cross(up, z_axis).normalized();
        auto y_axis = cross(z_axis, x_axis);
        view_matrix = Mat4f{
            Vec4f{x_axis, position[0]},
            Vec4f{y_axis, position[1]},
            Vec4f{z_axis, position[2]},
            Vec4f{0, 0, 0, 1}
        };

        float recip = 1.f / (far_plane - near_plane);
        float cot = 1.f / std::tan(to_radian(fov / 2.f));
        proj_matrix[0][0] = cot;
        proj_matrix[1][1] = cot;
        proj_matrix[2][2] = far_plane * recip;
        proj_matrix[2][3] = -near_plane * far_plane * recip;
        proj_matrix[3][2] = 1;
    }

    Ray generate_ray(uint x, uint y);

    void scale(const Vec3f& s) {
        horizontal *= s[0];
        vertical *= s[1];
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

    Mat4f view_matrix;
    Mat4f proj_matrix;
};