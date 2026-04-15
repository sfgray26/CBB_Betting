#!/usr/bin/env python
"""
CLI wrapper for crystaldba/postgres-mcp.

Spawns the postgres-mcp Docker container, sends a single tool request,
and prints the JSON result. This gives Gemini command-line access to
powerful Postgres MCP tools without needing an MCP client.

Requires: venv_mcp (created at project root with `pip install mcp`)

Usage:
    venv_mcp\\Scripts\\python scripts/devops/postgres_mcp_cli.py execute_sql "SELECT COUNT(*) FROM player_id_mapping"
    venv_mcp\\Scripts\\python scripts/devops/postgres_mcp_cli.py analyze_db_health
    venv_mcp\\Scripts\\python scripts/devops/postgres_mcp_cli.py list_schemas
    venv_mcp\\Scripts\\python scripts/devops/postgres_mcp_cli.py list_objects --schema public
    venv_mcp\\Scripts\\python scripts/devops/postgres_mcp_cli.py get_object_details --schema public --name player_id_mapping
    venv_mcp\\Scripts\\python scripts/devops/postgres_mcp_cli.py explain_query --query "SELECT * FROM player_id_mapping"
    venv_mcp\\Scripts\\python scripts/devops/postgres_mcp_cli.py get_top_queries --limit 5

Environment:
    DATABASE_URI is read from the environment. If not set, a default
    Railway public proxy URL is used.
"""
import argparse
import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

DEFAULT_DATABASE_URI = (
    "postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB"
    "@junction.proxy.rlwy.net:45402/railway"
)


async def _run_tool(tool_name: str, arguments: dict, access_mode: str = "restricted"):
    database_uri = os.environ.get("DATABASE_URL") or os.environ.get("DATABASE_URI") or DEFAULT_DATABASE_URI

    server_params = StdioServerParameters(
        command="docker",
        args=[
            "run", "-i", "--rm",
            "-e", f"DATABASE_URI={database_uri}",
            "crystaldba/postgres-mcp",
            f"--access-mode={access_mode}",
        ],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return result


def main() -> int:
    parser = argparse.ArgumentParser(description="CLI wrapper for postgres-mcp")
    parser.add_argument("tool", help="MCP tool name to call")
    parser.add_argument("positional", nargs="*", help="Positional arguments for the tool")
    parser.add_argument("--schema", help="Schema name (for list_objects, get_object_details)")
    parser.add_argument("--name", help="Object name (for get_object_details)")
    parser.add_argument("--query", help="SQL query (for explain_query)")
    parser.add_argument("--limit", type=int, help="Limit (for get_top_queries)")
    parser.add_argument("--access-mode", default="restricted", choices=["restricted", "unrestricted"],
                        help="Postgres MCP access mode")
    args = parser.parse_args()

    arguments: dict = {}

    if args.tool == "execute_sql":
        arguments["sql"] = " ".join(args.positional) if args.positional else ""
    elif args.tool == "explain_query":
        arguments["query"] = args.query or " ".join(args.positional)
    elif args.tool == "list_objects":
        arguments["schema"] = args.schema or (args.positional[0] if args.positional else "public")
    elif args.tool == "get_object_details":
        arguments["schema"] = args.schema or "public"
        arguments["name"] = args.name or (args.positional[0] if args.positional else "")
    elif args.tool == "get_top_queries":
        arguments["limit"] = args.limit or 10
        arguments["sort_by"] = "mean_time"
    elif args.tool == "analyze_db_health":
        pass
    elif args.tool == "list_schemas":
        pass
    elif args.tool == "analyze_query_indexes":
        arguments["queries"] = [args.query or " ".join(args.positional)]
    elif args.tool == "analyze_workload_indexes":
        pass
    else:
        if args.positional:
            arguments["input"] = " ".join(args.positional)

    try:
        result = asyncio.run(_run_tool(args.tool, arguments, access_mode=args.access_mode))
        # Convert MCP result to plain JSON for shell consumption
        output = {
            "isError": result.isError,
            "content": [
                {"type": c.type, "text": c.text}
                for c in result.content
            ],
        }
        print(json.dumps(output, indent=2, default=str))
        return 0 if not result.isError else 1
    except Exception as exc:
        print(json.dumps({"isError": True, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
