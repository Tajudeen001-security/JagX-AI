# JagX AI — System Scope

JagX AI is a general-purpose frontier-model research project, not a gaming-only model.

## Core capability families

1. General knowledge and question answering
2. Reasoning and mathematics
3. Software engineering
4. Complex web applications and APIs
5. UI/UX and highly animated web experiences
6. Game development across engines and frameworks
7. 2D, 3D, realistic and stylized game systems
8. Security engineering and defensive security analysis
9. Tool use and controlled computer interaction
10. Document understanding and transformation
11. Research, planning and long-running engineering tasks

## Game-engine independence

Godot is the first integration target because it is accessible for testing, but the architecture must not encode Godot-specific assumptions into the core model.

Planned adapters include:
- Godot
- Unreal Engine
- Unity
- Generic custom engines
- Web/game technologies such as WebGL/WebGPU

The game agent should reason about game concepts (scene graphs, entities, components, physics, rendering, animation, input, networking, assets and builds) and then translate those concepts into the selected engine.

## Web-development scope

JagX AI should eventually handle:
- frontend applications
- backend services
- databases
- authentication
- APIs
- real-time systems
- responsive UI
- animations
- WebGL/WebGPU experiences
- testing and deployment workflows

## Security scope

Security capability is intended for defensive engineering:
- secure code generation
- vulnerability identification
- dependency analysis
- threat modeling
- secure architecture
- sandboxed security testing
- remediation and regression testing

High-impact or destructive actions must remain behind explicit permissions and isolated environments.

## Frontier objective

The long-term objective is a highly capable general model and agent. Capability is measured by reproducible benchmarks rather than by marketing claims. Model scale will increase only when data quality, training stability, evaluation and available compute justify it.
