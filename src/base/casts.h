#pragma once

// Copied from applessed to do type cast like float->int
template <typename Target, typename Source>
inline Target binary_cast(Source s)
{
    static_assert(
        sizeof(Target) == sizeof(Source),
        "foundation::binary_cast() expects the source and target types to have the same size");

    union
    {
        Source  m_source;
        Target  m_target;
    } u;

    u.m_source = s;
    return u.m_target;
}