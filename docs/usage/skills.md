# Skills

OpenJet skills are filesystem instructions that the agent can discover, list, validate, and load on demand.

## Locations

Project skills:

```text
.openjet/skills/
.agents/skills/
```

User skills:

```text
~/.openjet/skills/
~/.agents/skills/
```

Bundled skills:

```text
<install>/skills/
```

Project skills override user and bundled skills with the same name. Prefer `.openjet/skills/` for OpenJet-native project skills. `.agents/skills/` is supported for compatibility with Agent Skills layouts.

## Standard Skill Layout

```text
.openjet/skills/my-skill/
  SKILL.md
  references/
  scripts/
  assets/
```

`SKILL.md` contains YAML frontmatter plus the skill instructions.

## CLI

```bash
openjet skill list
openjet skill view <name> [file_path]
openjet skill create <name>
openjet skill validate <name>
openjet skill doctor
```

`openjet skill create <name>` writes a standard skill directory under `.openjet/skills/`.

## Agent Tools

The runtime exposes two read-only tools for progressive skill loading:

- `skills_list` returns compact metadata for available skills.
- `skill_view` returns the full `SKILL.md`, or a file under a standard skill directory.
