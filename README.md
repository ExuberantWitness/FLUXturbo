# FLUXturbo — Flux-Insight Claim Chain v2 Blueprint MCP Server

MCP server that extracts Flux-Insight Claim Chain v2 knowledge graphs from academic text using LLM.

## CC Type System

### 8 Atom Types
`method` `bottleneck` `paper` `fact` `component` `hypothesis` `experiment` `verification`

### 11 Edge Types
| Type | Strength | Requires Rho |
|------|----------|-------------|
| EXTENDS | Strong causal | Yes |
| IMPROVES | Strong causal | Yes |
| REPLACES | Strong causal | Yes |
| ADAPTS | Strong causal | Yes |
| USES_COMPONENT | Weak | No |
| COMPARES | Weak | No |
| BACKGROUND | Semantic | No |
| IMPLEMENTS | Semantic | No |
| VALIDATES | Semantic | No |
| BOUNDARY_OF | Semantic | No |
| RELATED_TO | Semantic | No |

### Rho Evidence Record (strong causal edges)
`bottleneck` + `mechanism` + `tradeoff` + `confidence` (0.0–1.0)

### 14 Bottleneck Categories
`overestimation_bias` `training_instability` `sample_inefficiency` `exploration_exploitation` `credit_assignment` `catastrophic_forgetting` `scalability` `communication_overhead` `non_stationarity` `partial_observability` `multi_objective_conflict` `representational_limitation` `computational_cost` `generalization_gap`

## MCP Tools

- **extract_blueprint** — Extract CC atoms + edges from a single text segment
- **extract_batch** — Process multiple segments in parallel
- **get_cc_schema** — Return the full CC v2 ontology
- **validate_blueprint** — Validate atoms + edges against OntologyGatekeeper

## Installation

```bash
pip install mcp openai httpx
```

## Configuration (MCP client)

```json
{
  "mcpServers": {
    "cc-blueprint": {
      "command": "python",
      "args": ["-m", "cc_blueprint"],
      "cwd": "/path/to/FLUXturbo",
      "env": {
        "CC_API_KEY": "sk-your-deepseek-key",
        "CC_HTTP_PROXY": "http://127.0.0.1:6789",
        "CC_LLM_MODEL": "deepseek-chat",
        "CC_LLM_BASE_URL": "https://api.deepseek.com"
      }
    }
  }
}
```

## License

MIT
