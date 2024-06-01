#pragma once


#if defined(__CUDACC__) || defined(__CUDABE__)
#   define kazen_hostdevice __host__ __device__
#   define kazen_inline __forceinline__
#   define CONST_STATIC_INIT( ... )
#   define kazen_device __device__
#   define kazen_host   __host__
#   define kazen_align(n) __align__(n)
#else
#   define kazen_hostdevice 
#   define kazen_inline inline
#   define CONST_STATIC_INIT( ... ) = __VA_ARGS__
#   define kazen_device
#   define kazen_host
#   define kazen_align(n)
#endif