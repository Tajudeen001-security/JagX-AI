# JagX Instruction and Agent Training

This stage teaches behavior above base language modeling.

Training examples will use a structured task format:

- system constraints
- user goal
- relevant context
- available tools and permissions
- plan
- tool actions
- observations
- final verified result

Target behaviors include question answering, reasoning, coding, repository editing, app construction, real-time systems, web engineering, game engineering and defensive security.

Training data must distinguish proposed actions from verified outcomes. The model must not learn to claim that a build, test, deployment or security check succeeded without an observation proving it.
