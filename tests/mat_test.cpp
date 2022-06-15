

#include "base/mat.h"
#include "base/utils.h"
#include "core/sampling.h"

#include <iostream>
#include <memory>

int main()
{
    /*
    Mat2f mat2{3, 4, 5, 6};
    auto mat2_rev = mat2.inverse();

    std::cout << "mat2 : \n" << mat2 << std::endl;
    std::cout << "mat2 inverse : \n" << mat2_rev << std::endl;
    std::cout << mat2 * mat2_rev << std::endl;
    */

    Mat4f mat4{
        1, 1, 4, 5,
        3, 3, 3, 2,
        5, 1, 9, 0,
        9, 7, 7, 9};
    auto mat4_rev = mat4.inverse();

    std::cout << "mat4 : \n" << mat4 << std::endl;
    std::cout << "mat4 inverse : \n" << mat4_rev << std::endl;
    std::cout << mat4 * mat4_rev << std::endl;

    Mat4f mat4t{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        1, 2, 3, 1
    };
    Vec4f v4{0.f, 0.f, 0.f, 1.f};

    std::cout << "translate mat : " << mat4t << std::endl;
    std::cout << "vec mul : \n" << mat4t * v4 << std::endl;

    auto mat4i = Mat4f::identity();

    Mat4f mat_from_vec {
        Vec4f{1, 0, 0, 0},
        Vec4f{0, 1, 0, 0},
        Vec4f{0, 0, 1, 0},
        Vec4f{0, 0, 0, 1}
    };

    std::cout << "mat from vec : \n" << mat_from_vec << std::endl;

    auto aspect = 1.333333;
    auto far = 101.;
    auto near = 1.;
    auto recip = 1.f / 100.f;
    auto cot = 1.f / std::tan(to_radian(30. / 2.));
    auto trans = Mat4f::translate(Vec4f{-1.f, -1.f / aspect, 0.f, 1.f});
    auto scale = Mat4f::scale(Vec4f{2.f, 2.f / aspect, 1.f, 1.f});
    auto proj = Mat4f{
        cot, 0, 0, 0,
        0, cot, 0, 0,
        0, 0, far * recip, -near * far * recip,
        0, 0, 1, 0
    };

    auto comb = proj.inverse() * trans * scale;
    auto sample = random2f();

    std::cout << "proj mat : \n" << proj << std::endl;
    std::cout << "inv proj mat : \n" << proj.inverse() << std::endl;
    std::cout << "trans : \n" << trans << std::endl;
    std::cout << "with trans : \n" << proj.inverse() * trans << std::endl;
    std::cout << "scaled : \n" << scale * Vec4f{sample, 0.f, 1.f} << std::endl;
    std::cout << "scale and trans ed : \n" << trans * scale * Vec4f{sample, 0.f, 1.f} << std::endl;
    std::cout << "sampled : \n" << comb * Vec4f{sample, 0.f, 1.f} << std::endl;
    std::cout << "pos : \n" << comb * Vec4f{0.f, 0.f, 0.f, 1.f} << std::endl;

    auto position = Vec3f{0, 1, 0};
    auto lookat = Vec3f{1, 1, -1};
    auto up = Vec3f{0, 1, 0};
    auto z_axis = (lookat - position).normalized();
    auto x_axis = cross(z_axis, up).normalized();
    auto y_axis = cross(x_axis, z_axis);
    auto view_mat = Mat4f{
        Vec4f{x_axis, 0},
        Vec4f{y_axis, 0},
        Vec4f{z_axis, 0},
        Vec4f{position, 1}
    };
    auto pt = Vec4f{0, 0, -1, 1};
    std::cout << "view mat : \n" << view_mat << std::endl;
    std::cout << "cam to world : \n" << view_mat * pt << std::endl;

    return 0;
}