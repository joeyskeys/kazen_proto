

#include "base/mat.h"

#include <iostream>

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
}