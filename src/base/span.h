#pragma once

#include <array>
#include <limits>
#include <type_traits>

/*****
 * A practice of writting a lib.
 * Emulate std::span in cpp17
 *****/

namespace kp
{

template <typename T, std::size_t Extent=std::numeric_limits<std::size_t>::max()>
class span {
public:
    // types
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = std::size_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    //using iterator = ;

    // methods

    // ctors
    template <typename ET, std::size_t N, typename = std::enable_if_t<std::is_convertible_v<std::remove_cv_t<ET>, value_type>>>
    span(ET[N] arr) noexcept
        :data(arr)
    {
        // SFINAE out rather than error produced by static_assert
        //static_assert(std::is_convertible_v<std::remove_cv_t<ET>, value_type>);
        constexpr size = N;
    }

    template <typename ET, typename = std::enalbe_if_t<std::is_convertible_v<ET, T>>>>
    span(ET* arr_ptr, std::size_t s) noexcept
        : data(arr_ptr)
        , size(s)
    {
        static_assert(std::is_convertible_v<ET, T>);
        data = arr_ptr;
        size = s;
    }

    template <typename ET, std::size_t N, typename = std::enable_if_t<std::is_convertible_v<std::remove_cv_t<ET>, value_type>>>
    span(std::array<ET, N> arr) noexcept
        : data(arr.data())
    {
        constexpr size = N;
    }

    ~span() noexcept = default;

private:
    static constexpr std::size_t extend = Extent;
    pointer data;
    std::size_t size;
};

}