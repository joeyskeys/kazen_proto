#pragma once

enum ClosureID {
    // Referenced from appleseed

    // BSDF closures
    AshikhminShirleyID,
    BlinnID,
    DiffuseID,
    DisneyID,
    OrenNayarID,
    PhongID,
    ReflectionID,
    SheenID,
    HairID,
    TranslucentID,

    // Microfacet closures
    GlossyID,
    MetalID,
    GlassID,
    PlasticID,

    // Emission closures
    EmissionID,

    NumClosureIDs
};

