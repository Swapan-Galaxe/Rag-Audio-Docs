from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from batch_processor import BatchRAGProcessor
import logging

app = FastAPI(title="Batch RAG API")

# Enable CORS for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processors = {}
logger = logging.getLogger(__name__)

class ProcessRequest(BaseModel):
    directory: str
    batch_size: int = 100

class QueryRequest(BaseModel):
    processor_id: str
    question: str

@app.post("/process")
async def process_files(request: ProcessRequest):
    try:
        processor_id = f"batch_{request.directory.replace('/', '_')}"
        
        processor = BatchRAGProcessor(persist_directory=f"./vectordb_{processor_id}")
        
        # Load documents
        docs = processor.load_directory(request.directory)
        
        # Load audio files
        audio_docs = processor.process_audio_batch(request.directory.replace('documents', 'audio'))
        
        # Combine all documents
        all_docs = docs + audio_docs
        
        chunks = processor.process_in_batches(all_docs, request.batch_size)
        processor.create_vectorstore(chunks)
        
        processors[processor_id] = processor
        stats = processor.get_stats()
        
        return {
            "processor_id": processor_id,
            "status": "success",
            "stats": stats,
            "documents": len(docs),
            "audio_files": len(audio_docs)
        }
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        raise HTTPException(500, str(e))

@app.post("/query")
async def query(request: QueryRequest):
    try:
        if request.processor_id not in processors:
            raise HTTPException(404, "Processor not found")
        
        processor = processors[request.processor_id]
        answer = processor.query(request.question)
        
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error querying: {e}")
        raise HTTPException(500, str(e))

@app.get("/processors")
async def list_processors():
    return {"processors": list(processors.keys())}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
