Stage:
- verify

Rules:
- Verify the changed behavior before treating the task as complete.
- Prefer the narrowest meaningful test, lint, typecheck, or structural check.
- If verification fails, localize the failure instead of broadening scope immediately.
- If verification cannot run, state the exact blocker and the next concrete command to try.
