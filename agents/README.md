# JagX Agent Architecture

JagX is designed as a model + tool system. The language model plans and reasons; capability adapters perform bounded actions.

## Request lifecycle

1. Detect the user's language and requested output language.
2. Classify the task (chat, coding, research, media, game development, etc.).
3. Build a tool plan.
4. Request only the permissions required by that plan.
5. Execute inside a bounded workspace/sandbox where possible.
6. Inspect the result and run tests.
7. Repair failures iteratively.
8. Return the artifact and a concise explanation.

## Language behavior

English is the default, but output language is user-controlled. If a user asks for a Yoruba website, for example, the agent should generate the site's user-facing copy in Yoruba while keeping code syntax and required framework conventions valid. The same principle applies to other supported African languages.

## Engine adapters

Every engine adapter should expose the same high-level operations:

- `inspect`
- `scaffold`
- `edit`
- `run`
- `test`
- `debug`
- `export`

The adapter decides how these operations map to Godot, Unreal Engine, Unity or another development environment.

## Computer control

The registry is intentionally permissioned. Reading project files is different from executing arbitrary programs; executing a program is different from deleting data or performing external actions. Production integrations must preserve those boundaries and provide auditable tool calls.
