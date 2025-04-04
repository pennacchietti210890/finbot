import asyncio
import json

async def send_mcp_message(proc, message: dict):
    proc.stdin.write((json.dumps(message) + "\n").encode())
    await proc.stdin.drain()
    line = await proc.stdout.readline()
    return json.loads(line)

async def main():
    # Start your MCP server manually
    proc = await asyncio.create_subprocess_exec(
        "python", "app/finbot/mcp_finbot/mcp_servers_finbot.py",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    # Send initialize
    init_msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"roots": {"listChanged": True}},
            "clientInfo": {"name": "custom-client", "version": "0.1.0"}
        }
    }
    print("🚀 Sending initialize")
    init_resp = await send_mcp_message(proc, init_msg)
    print("✅ init response:", init_resp)

    # Send tools/list
    list_msg = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    }
    print("🧰 Sending tools/list")
    list_resp = await send_mcp_message(proc, list_msg)
    print("🧪 tools/list response:", json.dumps(list_resp, indent=2))

    # Send tools/call
    call_msg = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "tool_name": "get_historical_prices",
            "arguments": {
                "ticker": "AAPL",
                "day": "2",
            }
        }
    }
    print("📞 Calling tool 'get_historical_prices'")
    call_resp = await send_mcp_message(proc, call_msg)
    print("🗣️ tool call response:", call_resp)
    
    
    # Clean up
    proc.terminate()
    await proc.wait()

asyncio.run(main())