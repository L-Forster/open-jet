Keep work split into small turns.

Always optimize for:
- low RAM pressure
- bounded context windows
- narrow file scope
- resumable progress across turns

Rules:
- Work on the active step only.
- If a step becomes too broad, split it before continuing.
- Prefer exact file excerpts over broad repo dumps.
- Prefer dedicated tools over shell when both can solve the task.
- Keep outputs concise and operational.
- Preserve enough state that the next turn can continue with minimal context.

When context is tight:
- source beats skills
- skills beat memory
- memory beats old transcript

Never assume long chat history will remain available.
