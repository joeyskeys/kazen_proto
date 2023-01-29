#include "binding/utils.h"
#include "shading/bsdfs.h"

template <typename BSDF>
py::class_<BSDF> bind_bsdf(py::module& m, const char* name) {
    py::class_<BSDF> pycl(m, name);

    pycl.def_static("eval", &BSDF::eval, "evaluate the bsdf")
        .def_static("sample", &BSDF::sample, "sample the bsdf");

    return pycl;
}

void bind_bsdfs(py::module_& m) {
    py::module bsdfs = m.def_submodule("bsdfs",
        "BSDF functions");

    bind_bsdf<Diffuse>(bsdfs, "Diffuse");
    bind_bsdf<Phong>(bsdfs, "Phong");
    bind_bsdf<OrenNayar>(bsdfs, "OrenNayar");
    bind_bsdf<Ward>(bsdfs, "Ward");
    bind_bsdf<Reflection>(bsdfs, "Reflection");
    bind_bsdf<Refraction>(bsdfs, "Refraction");
    bind_bsdf<Transparent>(bsdfs, "Transparent");
    bind_bsdf<Translucent>(bsdfs, "Translucent");
    bind_bsdf<Emission>(bsdfs, "Emission");
    bind_bsdf<Background>(bsdfs, "Background");
    bind_bsdf<KpMirror>(bsdfs, "KpMirror");
    bind_bsdf<KpDielectric>(bsdfs, "KpDielectric");
    bind_bsdf<KpMicrofacet>(bsdfs, "KpMicrofacet");
    bind_bsdf<KpEmitter>(bsdfs, "KpEmitter");
    bind_bsdf<KpGloss>(bsdfs, "KpGloss");
    bind_bsdf<KpGlass>(bsdfs, "KpGlass");
    bind_bsdf<KpPrincipleDiffuse>(bsdfs, "KpPrincipleDiffuse");
    bind_bsdf<KpPrincipleRetro>(bsdfs, "KpPrincipleRetro");
    bind_bsdf<KpPrincipleFakeSS>(bsdfs, "KpPrincipleFakeSS");
    bind_bsdf<KpPrincipleSheen>(bsdfs, "KpPrincipleSheen");
    bind_bsdf<KpPrincipleSpecularReflection>(bsdfs, "KpPrincipleSpecularReflection");
    bind_bsdf<KpPrincipleClearcoat>(bsdfs, "KpPrincipleClearcoat");
}