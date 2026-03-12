---
tags:
  - tests
  - verification
  - pytest
mode: code
---
Use this skill when the active step needs verification.

Guidelines:
- prefer the narrowest test that catches the changed behavior
- avoid broad suites when a focused check is enough
- if tests are too expensive on-device, fall back to a smaller structural check
