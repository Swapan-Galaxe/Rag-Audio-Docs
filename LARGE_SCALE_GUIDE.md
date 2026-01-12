# Processing Large Datasets (Thousands of Files)

## Overview

All processors now include production-grade features:
- ✅ **Persistent Vector Store** - Data survives restarts
- ✅ **Logging** - Better debugging and monitoring
- ✅ **LangSmith Tracing** - Track LLM calls
- ✅ **Retry Logic** - Handles API failures (3 attempts with exponential backoff)
- ✅ **MMR Retrieval** - Maximum Marginal Relevance for diverse results
- ✅ **Error Handling** - Continues on file errors

## Strategies

### 1. Batch Processing (batch_processor.py)
Best for: Sequential processing with memory management

```python
from batch_processor import BatchRAGProcessor

processor = BatchRAGProcessor(persist_directory="./vectordb")

# Load all files from directory
docs = processor.load_directory("./data", file_types=["*.pdf", "*.txt", "*.docx"])

# Process in batches (adjust batch_size based on memory)
chunks = processor.process_in_batches(docs, batch_size=100)

# Create persistent vector store with batching
processor.create_vectorstore(chunks, batch_size=500)

# Query with MMR retrieval
answer = processor.query("What are the main topics?")
print(answer)

# Get stats
print(processor.get_stats())
```

**Features:**
- Persistent storage (survives restarts)
- Memory-efficient batching
- Progress bars with tqdm
- Automatic error recovery with retry logic
- MMR retrieval for diverse results

### 2. Parallel Processing (parallel_processor.py)
Best for: Fast processing with multiple CPU cores

```python
from parallel_processor import ParallelRAGProcessor

processor = ParallelRAGProcessor(
    persist_directory="./vectordb",
    max_workers=4  # Adjust based on CPU cores
)

# Process entire directory in parallel
chunks = processor.process_documents("./data", batch_size=500)

# Query with MMR retrieval
answer = processor.query("Summarize the key findings")
print(answer)
```

**Features:**
- Multi-threaded file loading (4 workers default)
- Persistent vector store
- Faster than batch processor
- Retry logic for API failures
- MMR retrieval

### 3. Incremental Processing (incremental_processor.py)
Best for: Adding new files without reprocessing everything

```python
from incremental_processor import IncrementalProcessor

processor = IncrementalProcessor(persist_directory="./vectordb")

# First run: processes all files
processor.process_new_files("./data")

# Add more files to ./data directory
# Second run: only processes new files
processor.process_new_files("./data")

# Query
answer = processor.query("What changed recently?")
print(answer)
```

**Features:**
- Tracks processed files in JSON log
- Only processes new/unprocessed files
- Persistent vector store
- Timestamps each file
- Inherits all improvements from BatchRAGProcessor

## Performance Tips

### Memory Optimization
- Reduce `chunk_size` (default: 1000)
- Reduce `batch_size` (default: 100-500)
- Process fewer files at once

```python
# Low memory settings
chunks = processor.process_in_batches(docs, batch_size=50)
processor.create_vectorstore(chunks, batch_size=100)
```

### Speed Optimization
- Use parallel processing
- Increase `max_workers`
- Use persistent vector store (no reprocessing)

```python
# Fast processing
processor = ParallelRAGProcessor(max_workers=8)
```

### Storage Optimization
- Use persistent vector store (automatic with persist_directory)
- Incremental processing for new files only
- MMR retrieval reduces redundant results

## Monitoring & Debugging

### Logging
All processors use Python's logging module:
```python
import logging
logging.basicConfig(level=logging.INFO)  # or DEBUG for more details
```

### LangSmith Tracing
Track all LLM calls:
```bash
# Add to .env
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=rag-audio-docs
```

### Progress Bars
All processors show progress with tqdm:
- File loading progress
- Batch processing progress
- Vector store creation progress

## Error Handling

All processors include:
- **Try-except blocks** - Skip corrupted files
- **Retry logic** - 3 attempts with exponential backoff
- **Logging** - Detailed error messages
- **Graceful degradation** - Continue processing on errors

## Alternative Vector Stores for Scale

### Pinecone (Cloud, Serverless)
```python
from langchain_community.vectorstores import Pinecone
import pinecone

pinecone.init(api_key="your-key")
vectorstore = Pinecone.from_documents(chunks, embeddings, index_name="my-index")
```

### FAISS (Local, Fast)
```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
```

### Weaviate (Self-hosted/Cloud)
```python
from langchain_community.vectorstores import Weaviate

vectorstore = Weaviate.from_documents(chunks, embeddings, weaviate_url="http://localhost:8080")
```

## Recommended Settings by Scale

### Small (< 1,000 files)
```python
processor = BatchRAGProcessor()
chunks = processor.process_in_batches(docs, batch_size=100)
processor.create_vectorstore(chunks, batch_size=500)
```

### Medium (1,000 - 10,000 files)
```python
processor = ParallelRAGProcessor(max_workers=4)
chunks = processor.process_documents("./data", batch_size=500)
```

### Large (> 10,000 files)
```python
# Use incremental processing + persistent storage
processor = IncrementalProcessor()
processor.process_new_files("./data", batch_size=200)

# Or use FAISS for local processing
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)
```

## Audio Processing at Scale

```python
# Process audio files in batches with retry logic
audio_docs = processor.process_audio_batch("./audio_files")
chunks = processor.process_in_batches(audio_docs, batch_size=50)
processor.create_vectorstore(chunks)
```

**Note:** Audio transcription is slow (~30-60s per file). Use batch processing for multiple files.

## Best Practices Summary

1. **Use persistent vector store** - Set `persist_directory` parameter
2. **Enable logging** - Set logging level to INFO or DEBUG
3. **Enable LangSmith** - Add API key to .env for tracing
4. **Use incremental processing** - For continuous data ingestion
5. **Use parallel processing** - For one-time large batch processing
6. **Monitor progress** - Watch tqdm progress bars and logs
7. **Handle errors gracefully** - Retry logic is automatic
8. **Use MMR retrieval** - Get diverse results automatically
