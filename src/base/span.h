#pragma once

#include <array>
#include <iterator>
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
    using iterator = pointer;
    using reverse_iterator = std::reverse_iterator<iterator>;

    // methods

    // ctors
    template <typename ET, std::size_t N, typename = std::enable_if_t<std::is_convertible_v<std::remove_cv_t<ET>, value_type>>>
    constexpr span(ET[N] arr) noexcept
        : ptr(arr)
        , size(N)
    {
        // SFINAE out rather than error produced by static_assert
        //static_assert(std::is_convertible_v<std::remove_cv_t<ET>, value_type>);
    }

    template <typename ET, typename = std::enalbe_if_t<std::is_convertible_v<ET, T>>>>
    constexpr span(ET* arr_ptr, std::size_t s) noexcept
        : ptr(arr_ptr)
        , size(s)
    {}

    template <typename ET, std::size_t N, typename = std::enable_if_t<std::is_convertible_v<std::remove_cv_t<ET>, value_type>>>
    constexpr span(std::array<ET, N> arr) noexcept
        : ptr(arr.data())
        , size(N)
    {}

    ~span() noexcept = default;

    constexpr span(const span& b) {
        ptr = b.ptr;
        size = b.size;
    }

    constexpr auto& operator =(const span& b) {
        ptr = b.ptr;
        size = b.size;
        return *this;
    }

    // iterators
    constexpr iterator begin() {
        return ptr;
    }

    constexpr iterator end() {
        return ptr + size;
    }

    constexpr reverse_iterator rbegin() {
        return reverse_iterator(end());
    }

    constexpr reverse_iterator rend() {
        return reverse_iterator(begin());
    }

    // element access
    constexpr reference front() const {
        return *ptr;
    }

    constexpr reference back() const {
        return *(ptr + size - 1);
    }

    constexpr reference operator [](size_type idx) const {
        assert(idx >= 0 && idx < size);
        return *(ptr  + idx);
    }

    constexpr pointer data() const  noexcept {
        return ptr;
    }

    // observers
    constexpr size_type size() const noexcept {
        return size;
    }

    constexpr bool empty() const noexcept {
        return size == 0;
    }

private:
    static constexpr std::size_t extend = Extent;
    pointer ptr;
    std::size_t size;
};

}