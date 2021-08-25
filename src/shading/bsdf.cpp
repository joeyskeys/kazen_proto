
#include "bsdf.h"

enum ClosureID {
    // Just add a few basic closures for test first

    // BSDF closures
    DiffuseID,

    // Microfacet closures
    GlossyID,

    // Emission closures
    EmissionID,

    NumClosureIDs
};

class Diffuse : public BSDF {
    struct DiffuseParams {

    };
}l=;