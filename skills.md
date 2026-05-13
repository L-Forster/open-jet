# Skills

This file is an index only.
If a skill looks relevant to the current task, call skill_view with its name before following it.
Do not assume the short summary here contains the full instructions.
Do not rely on absolute filesystem paths in this index.

global_skills_dir: <install>/skills
project_skills_dir: .openjet/skills (present)
project_agent_skills_dir: .agents/skills
user_skills_dirs: ~/.openjet/skills, ~/.agents/skills
merge_policy: project skills overlay global skills with the same name.
skill_loading: progressive_disclosure via skill_view(name).

Available skills:
- name: find-skills
  dir: ~/.agents/skills
  source: user
  format: standard
  load_name: find-skills
  use: Helps users discover and install agent skills when they ask questions like "how do I do X", "find a skill for X", "is there a skill that can...", or express interest in extending capabilities. This skill should be used when the user is l...
- name: gstack
  dir: .agents/skills
  source: project
  format: standard
  load_name: gstack
  allowed_tools: Bash, Read, AskUserQuestion
  use: Fast headless browser for QA testing and site dogfooding. Navigate pages, interact with elements, verify state, diff before/after, take annotated screenshots, test responsive layouts, forms, uploads, dialogs, and capture bug evidence. Us...
- name: jetson-debug
  dir: .openjet/skills
  source: project
  format: legacy
  load_name: jetson-debug
  tags: jetson, memory, debug, runtime
  use: Use this skill for edge-device or Jetson debugging.
- name: python-refactor
  dir: .openjet/skills
  source: project
  format: legacy
  load_name: python-refactor
  tags: python, refactor, implementation
  use: Use this skill when changing Python structure without changing intended behavior.
- name: review-changes
  dir: .openjet/skills
  source: project
  format: legacy
  load_name: review-changes
  tags: review, regression, findings
  use: Use this skill for code review turns.
- name: write-tests
  dir: .openjet/skills
  source: project
  format: legacy
  load_name: write-tests
  tags: tests, verification, pytest
  use: Use this skill when the active step needs verification.
