
// Closures

closure color kp_mirror() BUILTIN;

closure color kp_dielectric(float int_ior, float ext_ior) BUILTIN;

closure color kp_microfacet(float alpha, float int_ior, float ext_ior, float kd) BUILTIN;

closure color kp_emitter(float albedo) BUILTIN;

closure color kp_gloss(normal N, float xalpha, float yalpha, float eta, float f) BUILTIN;

closure color kp_glass(normal N, float xalpha, float yalpha, float eta, float f) BUILTIN;
