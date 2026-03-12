# failures

Append-only recurring failure patterns and fixes.

## 2026-03-12

- Pattern: prompt quality degrades when too many instruction files are loaded on small windows.
- Fix: load one role file, at most one skill on small windows, and prefer source over auxiliary docs.
## 2026-03-12
- mode: chat
- step: Test
- pattern: verification failed
- detail: [stderr] /home/louis/open-jet/.venv/bin/python: No module named pytest
- command: cd /home/louis/open-jet && .venv/bin/python -m pytest tests/ -v
## 2026-03-12
- mode: chat
- step: Test
- pattern: verification failed
- detail: [stderr] /bin/sh: 1: .venv/bin/pip: not found
- command: cd /home/louis/open-jet && .venv/bin/pip install pytest
## 2026-03-12
- mode: chat
- step: Test
- pattern: verification failed
- detail: [stderr] /bin/sh: 1: source: not found
- command: cd /home/louis/open-jet && source .venv/bin/activate && pip install pytest
## 2026-03-12
- mode: chat
- step: Test
- pattern: verification failed
- detail: [stderr] /bin/sh: 1: .: cannot open .venv/bin/activate: No such file
- command: cd /home/louis/open-jet && . .venv/bin/activate && pip install pytest
## 2026-03-12
- mode: chat
- step: Test
- pattern: verification failed
- detail: [stderr] /bin/sh: 1: .: cannot open .venv/bin/activate: No such file
- command: cd /home/louis/open-jet && . .venv/bin/activate && pip install pytest
