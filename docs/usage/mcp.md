# MCP Client Support

OpenJet can act as an MCP client and register selected MCP server tools as normal OpenJet tools. MCP is disabled by default and the Python SDK is optional so air-gapped installs keep working without extra dependencies.

Install the optional SDK support when you want to connect servers:

```bash
pip install "open-jet[mcp]"
```

## Configuration

MCP uses its own YAML file because it configures external tools, not the model runtime. For a project, create:

```text
.openjet/mcp.yaml
```

User-wide MCP servers can live at:

```text
~/.openjet/mcp.yaml
```

Project config overrides user config when both define the same server name. The older `config.yaml` `mcp:` section is still read as a compatibility fallback.

Example `.openjet/mcp.yaml`:

```yaml
enabled: true
default_timeout_seconds: 30
servers:
  github:
    enabled: true
    transport: stdio
    command: uvx
    args: ["mcp-server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "${GITHUB_PERSONAL_ACCESS_TOKEN}"
    confirmation_required: true
    tools:
      include: ["list_issues", "get_pull_request"]
      exclude: []
```

Each discovered tool is registered as:

```text
mcp_<server_name>_<tool_name>
```

For example, `github/list_issues` becomes `mcp_github_list_issues`.

`tools.include` is a whitelist. If it is non-empty, only those tools are exposed. If `include` is empty, `tools.exclude` hides matching tools.

## Security Defaults

- MCP is disabled unless `mcp.enabled` is true.
- MCP tools require confirmation by default.
- Stdio servers run with argv-style commands, never `shell=True`.
- Environment placeholders like `${TOKEN}` expand only from the current process environment.
- Secret-looking env/header values are redacted in status output.
- Only connect MCP servers you trust. Server tool descriptions and outputs are untrusted input.

## CLI

```bash
openjet mcp list
openjet mcp test github
openjet mcp add-stdio filesystem -- npx -y @modelcontextprotocol/server-filesystem .
openjet mcp remove filesystem
```

`add-stdio` and `remove` write `.openjet/mcp.yaml` in the current project.

Inside the TUI:

```text
/mcp status
```

## Streamable HTTP

Streamable HTTP servers can be configured with `transport: http`:

```yaml
enabled: true
servers:
  local_docs:
    enabled: true
    transport: http
    url: "http://127.0.0.1:8000/mcp"
    headers:
      Authorization: "Bearer ${LOCAL_DOCS_TOKEN}"
    confirmation_required: true
    tools:
      include: ["search", "fetch"]
```

Prefer streamable HTTP for new HTTP servers. Legacy SSE is not configured by OpenJet yet.

## Troubleshooting

- `MCP Python SDK is not installed`: install `open-jet[mcp]` or `mcp`.
- `duplicate generated MCP tool name`: rename the server or narrow `tools.include`; OpenJet rejects ambiguous generated tool names.
- A failed server does not stop OpenJet startup. Use `openjet mcp test <server>` to see the connection error.
- If a token is empty after `${VAR}` expansion, export the environment variable before launching OpenJet.
