#pragma once

enum class ClosureID {
    // Referenced from appleseed

    // BSDF closures
    AshikhminShirleyID,
    BlinnID,
    DiffuseID,
    DisneyID,
    OrenNayarID,
    PhongID,

    // Microfacet closures
    GlossyID,
    MetalID,

    // Emission closures
    EmissionID,

    NumClosureIDs
};