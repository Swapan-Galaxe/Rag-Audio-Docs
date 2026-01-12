# Processing Large Datasets (Thousands of Files)

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

# Create vector store with batching
processor.create_vectorstore(chunks, batch_size=500)

# Query
answer = processor.query("What are the main topics?")
print(answer)
```

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

# Query
answer = processor.query("Summarize the key findings")
print(answer)
```

### 3. Incremental Processing (incremental_processor.py)
Best for: Adding new files without reprocessing everything

```python
from incremental_processor import IncrementalProcessor

processor = IncrementalProcessor(persist_directory="./vectordb")

# First run: processes all files
processor.process_new_files("./data")

# Subsequent runs: only processes new files
processor.process_new_files("./data")

# Query
answer = processor.query("What changed recently?")
print(answer)
```

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
- Use faster embedding models

```python
# Fast processing
processor = ParallelRAGProcessor(max_workers=8)
```

### Storage Optimization
- Use persistent vector store (Chroma with persist_directory)
- Compress embeddings
- Use smaller embedding models

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
# Use incremental processing + cloud vector store
processor = IncrementalProcessor()
processor.process_new_files("./data", batch_size=200)

# Or use FAISS for local processing
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)
```

## Audio Processing at Scale

```python
# Process audio files in batches
audio_docs = processor.process_audio_batch("./audio_files")
chunks = processor.process_in_batches(audio_docs, batch_size=50)
processor.create_vectorstore(chunks)
```

## Monitoring Progress

All processors use `tqdm` for progress bars:
- File loading progress
- Batch processing progress
- Vector store creation progress

## Error Handling

All processors include try-except blocks to:
- Skip corrupted files
- Continue processing on errors
- Log failed files
