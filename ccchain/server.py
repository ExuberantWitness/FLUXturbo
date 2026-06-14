"""MCP Server — thin wrapper delegating to ccchain SDK."""

from __future__ import annotations

from mcp.server import Server
from mcp.server.stdio import stdio_server

from ccchain import ingest, search, evaluate
from ccchain.config import Config
from ccchain.core.ontology import (
    ATOM_TYPES,
    CC_EDGE_TYPES,
    HIERARCHY_EDGES,
    LEVELS,
    LEVEL_ALIAS,
    STRONG_CAUSAL_EDGES,
    TYPE_COMPATIBILITY,
)

server = Server("ccchain")


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    if name == "build_cc_index":
        result, err = ingest(
            segments=arguments["segments"],
            source_pdf=arguments["source_pdf"],
        )
    elif name == "search_cc":
        result, err = search(
            query=arguments["query"],
            top_k=arguments.get("top_k", 10),
            level=arguments.get("level", "W4"),
        )
    elif name == "evaluate_novelty":
        result, err = evaluate(
            proposal_text=arguments["proposal_text"],
            domain=arguments.get("domain"),
        )
    elif name == "get_cc_schema":
        result = {
            "atom_types": ATOM_TYPES,
            "edge_types": CC_EDGE_TYPES,
            "hierarchy_edges": HIERARCHY_EDGES,
            "strong_causal_edges": list(STRONG_CAUSAL_EDGES),
            "levels": LEVELS,
            "level_aliases": LEVEL_ALIAS,
            "type_compatibility": {
                k: list(v) for k, v in TYPE_COMPATIBILITY.items()
            },
        }
        err = None
    else:
        raise ValueError(f"Unknown tool: {name}")

    if err:
        return {"error": err}
    return {"result": result}


def main():
    import asyncio
    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    asyncio.run(run())


if __name__ == "__main__":
    main()
