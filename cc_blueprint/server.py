"""FLUXturbo — Flux-Insight Claim Chain v2 aligned Blueprint MCP Server.

Exposes CC-aligned knowledge-graph extraction as MCP tools:
  - extract_blueprint: extract CC atoms + edges from text via LLM
  - get_cc_schema:     return the CC type system (atom types, edge types, etc.)
  - validate_blueprint: validate a given set of atoms + edges against the ontology
  - extract_batch:     process multiple segments in parallel

Usage:
  python -m cc_blueprint

Or configure as an MCP server in Claude Code / any MCP client:
  {
    "mcpServers": {
      "cc-blueprint": {
        "command": "C:\\Users\\zhang\\.conda\\envs\\FLUX\\python.exe",
        "args": ["-m", "cc_blueprint"],
        "cwd": "E:\\DATA\\vscode\\FLUXturbo",
        "env": {
          "CC_API_KEY": "sk-your-deepseek-key",
          "CC_HTTP_PROXY": "http://127.0.0.1:6789",
          "CC_LLM_MODEL": "deepseek-chat",
          "CC_LLM_BASE_URL": "https://api.deepseek.com"
        }
      }
    }
  }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server
from mcp.server.models import InitializationOptions

from .extractor import BlueprintExtractor
from .ontology import (
    ATOM_TYPES,
    BOTTLENECK_CATEGORIES,
    EDGE_COMPAT,
    EDGE_TYPES,
    STRONG_CAUSAL,
    Atom,
    Blueprint,
    Edge,
    OntologyGatekeeper,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cc-blueprint-mcp")

# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------
_LLM_MODEL = os.getenv("CC_LLM_MODEL", "deepseek-chat")
_LLM_BASE_URL = os.getenv("CC_LLM_BASE_URL", "https://api.deepseek.com")
_API_KEY = os.getenv("CC_API_KEY", os.getenv("OPENAI_API_KEY", ""))
_HTTP_PROXY = os.getenv("CC_HTTP_PROXY", os.getenv("HTTP_PROXY", ""))
_MAX_ATOMS = int(os.getenv("CC_MAX_ATOMS", "30"))
_MAX_EDGES = int(os.getenv("CC_MAX_EDGES", "20"))

# ---------------------------------------------------------------------------
# Singleton instances (lazy init for extractor)
# ---------------------------------------------------------------------------
_extractor: BlueprintExtractor | None = None
gatekeeper = OntologyGatekeeper()


def _get_extractor() -> BlueprintExtractor:
    global _extractor
    if _extractor is None:
        _extractor = BlueprintExtractor(
            llm_model_name=_LLM_MODEL,
            llm_base_url=_LLM_BASE_URL,
            api_key=_API_KEY,
            http_proxy=_HTTP_PROXY,
            max_atoms=_MAX_ATOMS,
            max_edges=_MAX_EDGES,
        )
    return _extractor

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
app = Server("cc-blueprint")


@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="extract_blueprint",
            description=(
                "Extract Flux-Insight CC atoms (method/bottleneck/paper/fact/component/"
                "hypothesis/experiment/verification) and CC edges (EXTENDS/IMPROVES/"
                "REPLACES/ADAPTS/USES_COMPONENT/COMPARES/BACKGROUND/IMPLEMENTS/VALIDATES/"
                "BOUNDARY_OF/RELATED_TO) from academic text using LLM. "
                "Returns validated atoms and edges with Rho evidence for strong-causal edges."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Academic text to extract knowledge graph from",
                    },
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="extract_batch",
            description=(
                "Extract CC blueprints from multiple text segments in parallel. "
                "Returns a list of blueprints, one per segment."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "segments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of text segments to process",
                    },
                    "max_workers": {
                        "type": "integer",
                        "description": "Parallel workers (default 4)",
                        "default": 4,
                    },
                },
                "required": ["segments"],
            },
        ),
        types.Tool(
            name="get_cc_schema",
            description=(
                "Return the Flux-Insight Claim Chain v2 ontology: "
                "8 atom types, 11 edge types, type-compatibility matrix, "
                "14 bottleneck categories, and strong-causal classification."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="validate_blueprint",
            description=(
                "Validate a set of CC atoms and edges against the OntologyGatekeeper. "
                "Checks: atom name validity, type validity, reference integrity (R1), "
                "Rho completeness for strong-causal edges (R4), type compatibility."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "atoms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "context": {"type": "string"},
                            },
                            "required": ["name", "type"],
                        },
                        "description": "List of CC atoms to validate",
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "src": {"type": "string"},
                                "relation": {"type": "string"},
                                "tgt": {"type": "string"},
                                "rho": {
                                    "type": "object",
                                    "properties": {
                                        "bottleneck": {"type": "string"},
                                        "mechanism": {"type": "string"},
                                        "tradeoff": {"type": "string"},
                                        "confidence": {"type": "number"},
                                    },
                                },
                            },
                            "required": ["src", "relation", "tgt"],
                        },
                        "description": "List of CC edges to validate",
                    },
                },
                "required": ["atoms", "edges"],
            },
        ),
    ]


@app.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    args = arguments or {}

    try:
        if name == "extract_blueprint":
            return await _handle_extract_blueprint(args)
        elif name == "extract_batch":
            return await _handle_extract_batch(args)
        elif name == "get_cc_schema":
            return await _handle_get_schema()
        elif name == "validate_blueprint":
            return await _handle_validate(args)
        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return [types.TextContent(type="text", text=f"Error: {exc}")]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------
async def _handle_extract_blueprint(args: dict) -> list[types.TextContent]:
    text = args["text"]
    loop = asyncio.get_event_loop()
    bp = await loop.run_in_executor(None, _get_extractor().extract_from_text, text)
    return [types.TextContent(type="text", text=json.dumps(bp.to_dict(), ensure_ascii=False, indent=2))]


async def _handle_extract_batch(args: dict) -> list[types.TextContent]:
    segments = args["segments"]
    max_workers = args.get("max_workers", 4)
    loop = asyncio.get_event_loop()
    bps = await loop.run_in_executor(
        None, _get_extractor().extract_batch, segments, max_workers
    )
    return [
        types.TextContent(
            type="text",
            text=json.dumps([bp.to_dict() for bp in bps], ensure_ascii=False, indent=2),
        )
    ]


async def _handle_get_schema() -> list[types.TextContent]:
    schema = {
        "atom_types": ATOM_TYPES,
        "edge_types": EDGE_TYPES,
        "strong_causal": sorted(STRONG_CAUSAL),
        "bottleneck_categories": BOTTLENECK_CATEGORIES,
        "edge_compatibility": {
            et: [f"{s}→{t}" for s, t in pairs]
            for et, pairs in EDGE_COMPAT.items()
            if pairs
        },
    }
    return [types.TextContent(type="text", text=json.dumps(schema, indent=2))]


async def _handle_validate(args: dict) -> list[types.TextContent]:
    atoms_data = args.get("atoms", [])
    edges_data = args.get("edges", [])

    atoms: list[Atom] = []
    for a in atoms_data:
        try:
            atoms.append(Atom.from_dict(a))
        except (ValueError, KeyError):
            atoms.append(Atom(name=str(a.get("name", "?")), type="method"))

    edges: list[Edge] = []
    for e in edges_data:
        try:
            edges.append(Edge.from_dict(e))
        except (ValueError, KeyError):
            pass

    bp = Blueprint(atoms=atoms, edges=edges)
    errors = gatekeeper.validate_blueprint(bp)

    result = {
        "valid": len(errors) == 0,
        "error_count": len(errors),
        "errors": errors,
        "atom_count": len(atoms),
        "edge_count": len(edges),
    }
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


# ---------------------------------------------------------------------------
# Server runner
# ---------------------------------------------------------------------------
def _init_options() -> InitializationOptions:
    return InitializationOptions(
        server_name="cc-blueprint",
        server_version="0.1.0",
        capabilities=types.ServerCapabilities(
            tools=types.ToolsCapability(),
        ),
    )


async def run_stdio():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, _init_options())


def main():
    """Entry point for `python -m cc_blueprint`."""
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    asyncio.run(run_stdio())


if __name__ == "__main__":
    main()
