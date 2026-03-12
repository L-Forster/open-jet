---
tags:
  - python
  - refactor
  - implementation
mode: code
---
Use this skill when changing Python structure without changing intended behavior.

Guidelines:
- inspect the current call path before editing
- preserve external behavior unless the step explicitly changes it
- prefer small, local refactors over broad rewrites
- keep imports, types, and control flow easy to verify
- run a focused syntax/import check after the change
