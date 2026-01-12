# MCP Server Setup

## Install MCP

```bash
pip install mcp
```

## Run MCP Server

```bash
py mcp_server.py
```

## Configure in Claude Desktop

Add to `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag-audio-docs": {
      "command": "py",
      "args": ["c:\\Users\\sdeep\\AppData\\Local\\Temp\\rag_audio_docs\\mcp_server.py"]
    }
  }
}
```

## Available Tools

1. **load_documents** - Load all documents and audio from data/ directory
2. **query_documents** - Ask questions about loaded content
3. **get_summary** - Get comprehensive summary of all content

## Usage in Claude Desktop

After configuration, Claude can use these tools:
- "Load the documents from the RAG system"
- "What are the main topics in the documents?"
- "Give me a summary of all the content"
