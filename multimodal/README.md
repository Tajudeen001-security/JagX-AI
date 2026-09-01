# JagX Multimodal Core

JagX treats multimodality as a first-class subsystem. A shared language/token representation is connected to replaceable modality encoders and generative decoders.

## Implemented foundation
- Vision feature projector into JagX token space
- Audio feature projector interface

## Roadmap
1. image understanding via vision encoder + projector
2. native image generation decoder
3. video temporal encoder
4. native short-video generation decoder
5. audio understanding/generation
6. joint multimodal instruction tuning

External encoders/decoders may be used as replaceable research components, but the core architecture has no mandatory third-party AI runtime dependency. Large model weights belong in artifact storage, not Git.
