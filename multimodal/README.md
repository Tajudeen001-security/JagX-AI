# JagX Multimodal Generation

JagX is designed to support image and video generation as first-class capabilities while keeping the core system provider-independent.

## Image generation
Target pipeline:
text/image conditioning → planning → generation model → safety checks → quality checks → artifact.

## Short video generation
Target pipeline:
prompt/storyboard → keyframes → temporal generation → consistency checks → encoding → artifact.

Short-form outputs should support configurable duration, aspect ratio, frame rate and resolution.

## Architecture
Generation models are replaceable modules. The assistant must not require a third-party AI API at runtime. A future native JagX generative stack can replace experimental backends without changing the agent interface.

Large generative models and weights are artifacts, not Git repository files.
