# Game Engine Adapters

JagX AI is engine-independent.

## Adapter contract

An engine adapter should expose:
- project discovery
- scene/project representation
- asset manifest
- code generation targets
- build command
- test command
- runtime log collection
- error parsing
- repair hooks

## Initial adapters

### Godot
First implementation target.

### Unreal Engine
Planned. The adapter will cover Unreal project structure, C++/Blueprint-oriented workflows, assets, packaging and automated validation.

### Unity
Planned. The adapter will cover Unity project structure, C#, scenes, prefabs, assets and builds.

### Generic
A generic adapter will support custom engines and web-native games.

The model should learn transferable game-development concepts instead of memorizing one engine.
