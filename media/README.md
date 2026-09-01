# JagX Media Generation

Media generation is structured as a controllable pipeline rather than a single opaque call.

## Movie pipeline
story → screenplay → shot list → scene assets → keyframes → temporal generation → audio → editing → continuity checks → export

The movie specification currently validates projects up to 20 minutes. Longer limits can be introduced after quality, memory and compute requirements are benchmarked.

Every generated asset should carry seed, model version, prompt/specification and provenance metadata so scenes can be regenerated consistently.
