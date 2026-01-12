import asyncio
from mcp.server import Server
from mcp.types import Tool, TextContent
from rag_summarizer import RAGSummarizer
from pathlib import Path

app = Server("rag-audio-docs")
summarizer = RAGSummarizer()
processed = False

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="load_documents",
            description="Load and process documents and audio files from the data directory",
            inputSchema={
                "type": "object",
                "properties": {},
            }
        ),
        Tool(
            name="query_documents",
            description="Ask questions about the loaded documents and audio files",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask about the documents"
                    }
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="get_summary",
            description="Get a comprehensive summary of all loaded documents and audio",
            inputSchema={
                "type": "object",
                "properties": {},
            }
        ),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    global processed
    
    if name == "load_documents":
        script_dir = Path(__file__).parent
        doc_dir = script_dir / "data" / "documents"
        audio_dir = script_dir / "data" / "audio"
        
        docs = list(doc_dir.glob("*.txt")) + list(doc_dir.glob("*.pdf"))
        audios = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav"))
        
        if not docs and not audios:
            return [TextContent(type="text", text="No files found in data/ directory")]
        
        summarizer.process_files(
            [str(d) for d in docs] if docs else None,
            [str(a) for a in audios] if audios else None
        )
        processed = True
        return [TextContent(
            type="text",
            text=f"Successfully loaded {len(docs)} documents and {len(audios)} audio files"
        )]
    
    elif name == "query_documents":
        if not processed:
            return [TextContent(type="text", text="Please load documents first using load_documents tool")]
        
        question = arguments["question"]
        answer = summarizer.query(question)
        return [TextContent(type="text", text=answer)]
    
    elif name == "get_summary":
        if not processed:
            return [TextContent(type="text", text="Please load documents first using load_documents tool")]
        
        summary = summarizer.custom_summary(
            "Provide a comprehensive summary covering all topics from all documents and audio"
        )
        return [TextContent(type="text", text=summary)]
    
    return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
