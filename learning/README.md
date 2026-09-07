# JagX Learning Layer

JagX separates **learning from uncontrolled self-modification**.

The runtime can collect approved interaction feedback, task outcomes, language corrections, and domain examples into versioned datasets. Training jobs can then use those datasets to produce auditable checkpoints.

The production agent should never silently rewrite its own model weights. New checkpoints must be evaluated, versioned, and promoted by a controlled deployment process.

Recommended feedback record fields:
- task_id
- input_language
- task_type
- anonymized input/output hashes
- user rating or correction
- tool outcome
- safety outcome
- dataset/version provenance
