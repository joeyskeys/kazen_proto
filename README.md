# kazen_proto

###  LOG:

- [ ] Add Dual support to have dPdx, dPdu, dudx etc...
- [x] Have texture rendered after shaderglobal is complete(Update : texture support is there after setup u&v in the shaderglobal, no need for dPdx and etc.. Texture support is done now)
- [x] easier class introspection with boost hana and frozen. Refer to https://github.com/nicktrandafil/yenxo to get runtime support (Update : tested boost hana but no good result, cannot save much typing with a lot extra coding difficulties and other flaws, check branch with_hana)
- [x] configurable light path record for debugging and visualization (Update : not configurable now with hard coded structure, can provide convinience for some basic debug. Add other plans on this topic later on)
- [ ] Complete closures and fill up with details of the various microfacet models
- [ ] look into path integrator to quantify the errors between ground truth

### Kazen Con 2022.3.1 Briefing

  1. Current development status of kazen proto and related background knowledge(OIIO, OSL).
  2. Design and ideas of some module of the renderer:
      1. Scene description and parsing;
      2. Kernel seperation of OSL for future integration of GPU;
      3. Debugging tools;
      4. Testing, benchmarking and documenting.
  3. Problems occurred in development of kazen proto and related topics that need further investigation:
      1. floating point and self intersection;
      2. cpp self introspection;
      3. partial derivatives and dual number.
  4. Discussion and quick revisit of light path transport, importance sampling, MIS etc..