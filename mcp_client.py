"""
mcp_client.py

A command-line client that connects to a local MCP server, retrieves
available tools, and interacts with a local Ollama model to execute them.
"""

import asyncio
import os
import sys
from typing import List, Dict, Any

import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Define the local model to use. 
OLLAMA_MODEL = "qwen3.5:9b"


async def chat_loop(session: ClientSession, ollama_tools: List[Dict[str, Any]]) -> None:
    """
    Run the interactive chat loop, sending user input to Ollama and handling tool calls.
    
    Args:
        session (ClientSession): The active MCP client session.
        ollama_tools (list): A list of tool schemas formatted for Ollama.
    """
    messages = []
    print("\n[Client] Ready! Type 'quit' or 'exit' to stop.")
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit']:
                break
                
            messages.append({"role": "user", "content": user_input})
            print(f"[Client] Querying {OLLAMA_MODEL}...")
            
            # Send the conversation and available tools to Ollama
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                tools=ollama_tools
            )
            
            msg = response.get("message", {})
            
            # Check if the model decided to call any tools
            if msg.get("tool_calls"):
                print(f"[Client] Model requested {len(msg['tool_calls'])} tool call(s).")
                messages.append(msg)
                
                # Execute each requested tool
                for tool_call in msg["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]
                    
                    print(f"  [Tool Execution] Calling '{tool_name}' with args: {tool_args}")
                    
                    try:
                        # Call the tool via the MCP session
                        result = await session.call_tool(tool_name, arguments=tool_args)
                        # Extract the text content from the result list
                        tool_result_text = "\n".join([c.text for c in result.content if c.type == "text"])
                        print(f"  [Tool Result] {tool_result_text}")
                        
                    except Exception as e:
                        tool_result_text = f"Error executing tool: {str(e)}"
                        print(f"  [Tool Error] {tool_result_text}")
                        
                    # Append the tool's result to the conversation history
                    messages.append({
                        "role": "tool",
                        "content": tool_result_text,
                        "name": tool_name
                    })
                    
                # Send the tool results back to Ollama to generate a final response
                print("[Client] Sending tool results back to the model...")
                response = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=messages
                )
                final_msg = response.get("message", {})
                messages.append(final_msg)
                print(f"\nAssistant: {final_msg.get('content')}")
                
            else:
                # Normal text response with no tools used
                messages.append(msg)
                print(f"\nAssistant: {msg.get('content')}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[Client] An error occurred: {e}")
            break


async def main() -> None:
    """
    Main entry point for the MCP client.
    """
    # Dynamically locate the server script in the same directory as this client
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(current_dir, "mcp_server.py")
    
    if not os.path.exists(server_script):
        print(f"[Error] Could not find server script at: {server_script}")
        print("Please ensure 'mcp_server.py' is in the directory.")
        sys.exit(1)
    
    # Configure the server parameters to run the script via the current Python executable
    server_params = StdioServerParameters(
        command=sys.executable,  # Uses the exact Python environment running the client
        args=[server_script],
        env=None
    )

    print(f"[Client] Connecting to MCP server at '{server_script}'...")
    
    # Establish the standard IO connection to the server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Retrieve the list of tools exposed by the server
            tools_response = await session.list_tools()
            mcp_tools = tools_response.tools
            
            # Format the MCP tools into the standard JSON schema expected by Ollama
            ollama_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                }
                for tool in mcp_tools
            ]
            
            print(f"[Client] Connected! Available tools: {[t.name for t in mcp_tools]}")
            
            # Start the interactive loop
            await chat_loop(session, ollama_tools)


if __name__ == "__main__":
    # Run the asynchronous main loop
    asyncio.run(main())