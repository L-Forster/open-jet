# failures

Append-only recurring failure patterns and fixes.

## 2026-03-12

- Pattern: prompt quality degrades when too many instruction files are loaded on small windows.
- Fix: load one role file, at most one skill on small windows, and prefer source over auxiliary docs.
