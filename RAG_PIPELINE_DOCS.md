# RAG Audio & Document Pipeline - Complete Documentation

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [File Structure](#file-structure)
4. [Technology Stack](#technology-stack)
5. [Data Flow](#data-flow)
6. [Best Practices Implemented](#best-practices-implemented)
7. [Deployment Options](#deployment-options)
8. [API Reference](#api-reference)

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interfaces                          │
├──────────────────┬──────────────────┬──────────────────────┤
│  Gradio Web UI   │   React Web UI   │   MCP Server         │
│  (chatbot_ui.py) │ (React App)      │  (mcp_server.py)     │
└────────┬─────────┴────────┬─────────┴──────────┬───────────┘
         │                  │                     │
         └──────────────────┼─────────────────────┘
                            │
                    ┌───────▼────────┐
                    │   FastAPI      │
                    │ (api_largescale)│
                    └───────┬────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
    ┌────▼─────┐    ┌──────▼──────┐   ┌──────▼──────┐
    │  Batch   │    │  Parallel   │   │ Incremental │
    │Processor │    │ Processor   │   │  Processor  │
    └────┬─────┘    └──────┬──────┘   └──────┬──────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            │
                    ┌───────▼────────┐
                    │ RAG Summarizer │
                    │  (Core Engine) │
                    └───────┬────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
    ┌────▼─────┐    ┌──────▼──────┐   ┌──────▼──────┐
    │Document  │    │   Whisper   │   │   OpenAI    │
    │ Loaders  │    │   (Audio)   │   │  (LLM/Emb)  │
    └────┬─────┘    └──────┬──────┘   └──────┬──────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            │
                    ┌───────▼────────┐
                    │ Chroma Vector  │
                    │   Database     │
                    └────────────────┘
```

---

## Core Components

### 1. RAG Summarizer (`rag_summarizer.py`)
**Purpose:** Core RAG engine for basic document processing

**Key Features:**
- Document loading (PDF, DOCX, TXT)
- Audio transcription (Whisper AI)
- Text chunking (1000 chars, 200 overlap)
- Vector embeddings (OpenAI)
- Persistent vector store (Chroma)
- MMR retrieval for diverse results
- Retry logic (3 attempts, exponential backoff)
- LangSmith tracing

**Methods:**
- `transcribe_audio()` - Convert audio to text
- `load_documents()` - Load various file formats
- `process_files()` - Main processing pipeline
- `query()` - RAG-based Q&A
- `custom_summary()` - Generate summaries

### 2. Batch Processor (`batch_processor.py`)
**Purpose:** Sequential processing for large datasets

**Key Features:**
- Memory-efficient batching
- Directory-level loading
- Progress bars (tqdm)
- Persistent storage
- Audio batch processing
- Error recovery

**Use Case:** 
- Processing thousands of files
- Memory-constrained environments
- Sequential workflows

### 3. Parallel Processor (`parallel_processor.py`)
**Purpose:** Fast multi-threaded processing

**Key Features:**
- ThreadPoolExecutor (4 workers default)
- Parallel file loading
- Faster than batch processor
- Same persistence features

**Use Case:**
- One-time large batch processing
- Multi-core systems
- Speed-critical scenarios

### 4. Incremental Processor (`incremental_processor.py`)
**Purpose:** Smart updates for continuous ingestion

**Key Features:**
- Tracks processed files (JSON log)
- Only processes new files
- Timestamps each file
- Inherits batch processor features

**Use Case:**
- Continuous data ingestion
- Avoiding reprocessing
- Incremental updates

---

## File Structure

```
rag_audio_docs/
├── Core Engine
│   ├── rag_summarizer.py          # Main RAG engine
│   ├── batch_processor.py         # Large-scale batch processing
│   ├── parallel_processor.py      # Multi-threaded processing
│   └── incremental_processor.py   # Incremental updates
│
├── User Interfaces
│   ├── chatbot_ui.py              # Gradio web UI
│   ├── api_largescale.py          # FastAPI backend
│   ├── mcp_server.py              # MCP server for AI assistants
│   └── react-largescale-ui/       # React web application
│       ├── src/
│       │   ├── App.js             # Main React component
│       │   ├── App.css            # Styling
│       │   └── index.js           # Entry point
│       ├── public/
│       │   └── index.html         # HTML template
│       └── package.json           # Dependencies
│
├── Data & Storage
│   ├── data/
│   │   ├── documents/             # PDF, TXT, DOCX files
│   │   └── audio/                 # MP3, WAV files
│   ├── chroma_db/                 # Vector database (persistent)
│   └── test_vectordb/             # Test vector storage
│
├── Configuration
│   ├── .env                       # API keys (not in repo)
│   ├── .env.example               # Template for .env
│   ├── requirements.txt           # Python dependencies
│   └── Dockerfile                 # Container image
│
├── Documentation
│   ├── README.md                  # Quick start guide
│   ├── LARGE_SCALE_GUIDE.md       # Large dataset processing
│   ├── AZURE_DEPLOYMENT.md        # Cloud deployment
│   ├── CICD_SETUP.md              # CI/CD pipeline
│   ├── MCP_SETUP.md               # MCP server setup
│   └── RAG_PIPELINE_DOCS.md       # This file
│
├── Deployment
│   ├── Dockerfile                 # Container definition
│   ├── deploy_azure.sh            # Azure deployment script
│   └── .github/workflows/
│       └── deploy.yml             # GitHub Actions CI/CD
│
└── Testing
    └── test_files.py              # Integration tests
```

---

## Technology Stack

### Backend
- **Python 3.10+**
- **LangChain** - RAG framework
  - `langchain-openai` - OpenAI integration
  - `langchain-community` - Document loaders, vector stores
  - `langchain-core` - Core abstractions
- **OpenAI** - LLM and embeddings
  - GPT-3.5-turbo for generation
  - text-embedding-ada-002 for embeddings
- **Whisper** - Audio transcription
- **Chroma** - Vector database
- **FastAPI** - REST API framework
- **Gradio** - Web UI framework

### Frontend
- **React 18** - UI framework
- **Axios** - HTTP client

### DevOps
- **Docker** - Containerization
- **Azure Container Apps** - Cloud hosting
- **GitHub Actions** - CI/CD

### Monitoring
- **LangSmith** - LLM tracing
- **Python Logging** - Application logs
- **Tenacity** - Retry logic

---

## Data Flow

### 1. Document Processing Flow

```
Input Files (PDF/DOCX/TXT/MP3)
    ↓
Document Loaders (PyPDF/Docx2txt/TextLoader/Whisper)
    ↓
Raw Text Documents
    ↓
Text Splitter (RecursiveCharacterTextSplitter)
    ↓
Text Chunks (1000 chars, 200 overlap)
    ↓
OpenAI Embeddings API
    ↓
Vector Embeddings (1536 dimensions)
    ↓
Chroma Vector Database (Persistent)
```

### 2. Query Flow

```
User Question
    ↓
OpenAI Embeddings API (Question → Vector)
    ↓
Chroma Vector Search (MMR, k=3-5)
    ↓
Retrieved Relevant Chunks
    ↓
Prompt Template (Context + Question)
    ↓
OpenAI GPT-3.5-turbo
    ↓
Generated Answer
    ↓
User
```

### 3. RAG Pipeline Steps

1. **Ingestion**
   - Load documents from various formats
   - Transcribe audio files
   - Extract text content

2. **Chunking**
   - Split into manageable pieces
   - Maintain context with overlap
   - Preserve metadata

3. **Embedding**
   - Convert text to vectors
   - Use OpenAI embeddings
   - Store in vector database

4. **Retrieval**
   - Semantic search on query
   - MMR for diversity
   - Return top-k chunks

5. **Generation**
   - Build prompt with context
   - Call LLM (GPT-3.5)
   - Parse and return answer

---

## Best Practices Implemented

### 1. Persistent Storage
- Vector store survives restarts
- Incremental updates supported
- No reprocessing needed

### 2. Error Handling
- Try-catch blocks everywhere
- Retry logic (3 attempts, exponential backoff)
- Graceful degradation
- Detailed error logging

### 3. Logging & Monitoring
- Python logging module
- LangSmith tracing for LLM calls
- Progress bars (tqdm)
- Performance metrics

### 4. Retrieval Strategy
- MMR (Maximum Marginal Relevance)
- Diverse results
- Configurable k parameter
- Fetch more, return best

### 5. Scalability
- Batch processing for memory efficiency
- Parallel processing for speed
- Incremental processing for updates
- Configurable batch sizes

### 6. Security
- Environment variables for secrets
- No hardcoded credentials
- CORS configuration
- Input validation

### 7. Code Quality
- Type hints (Pydantic models)
- Docstrings
- Modular design
- DRY principle

---

## Deployment Options

### 1. Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run Gradio UI
python chatbot_ui.py

# Run FastAPI + React
python api_largescale.py
cd react-largescale-ui && npm start
```

### 2. Docker
```bash
# Build image
docker build -t rag-chatbot .

# Run container
docker run -p 7860:7860 -e OPENAI_API_KEY=xxx rag-chatbot
```

### 3. Azure Container Apps
```bash
# Deploy
bash deploy_azure.sh

# Or use GitHub Actions CI/CD
git push origin main
```

### 4. MCP Server (AI Assistants)
```bash
# Run MCP server
python mcp_server.py

# Configure in Claude Desktop
# See MCP_SETUP.md
```

---

## API Reference

### FastAPI Endpoints

#### POST /process
Process files from directory

**Request:**
```json
{
  "directory": "./data/documents",
  "batch_size": 100
}
```

**Response:**
```json
{
  "processor_id": "batch_data_documents",
  "status": "success",
  "stats": "Vector store contains 150 chunks",
  "documents": 2,
  "audio_files": 1
}
```

#### POST /query
Query processed documents

**Request:**
```json
{
  "processor_id": "batch_data_documents",
  "question": "What are the main topics?"
}
```

**Response:**
```json
{
  "answer": "The main topics discussed are..."
}
```

#### GET /processors
List active processors

**Response:**
```json
{
  "processors": ["batch_data_documents"]
}
```

#### GET /health
Health check

**Response:**
```json
{
  "status": "healthy"
}
```

---

## Performance Metrics

### Processing Speed
- **TXT files:** ~0.1s per file
- **PDF files:** ~2-10s per file
- **Audio files:** ~30-60s per file (depends on length)

### Memory Usage
- **Batch size 100:** ~500MB RAM
- **Batch size 500:** ~2GB RAM

### Scalability
- **Small (<1,000 files):** Batch processor
- **Medium (1,000-10,000):** Parallel processor
- **Large (>10,000):** Incremental processor

---

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional (LangSmith tracing)
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=rag-audio-docs
```

---

## Supported File Formats

### Documents
- PDF (.pdf)
- Word (.docx)
- Text (.txt)

### Audio
- MP3 (.mp3)
- WAV (.wav)
- M4A (.m4a)
- Any format supported by Whisper

---

## Cost Estimation

### OpenAI API Costs (per 1000 files)
- **Embeddings:** ~$0.50-2.00
- **GPT-3.5 queries:** ~$0.10-0.50 per query
- **Whisper transcription:** ~$0.006 per minute

### Azure Hosting (monthly)
- **Container Apps:** ~$35-55
- **Container Registry:** ~$5
- **Total:** ~$40-60/month

---

## Troubleshooting

### Common Issues

1. **Import errors**
   - Run: `pip install -r requirements.txt`

2. **Audio transcription fails**
   - Check ffmpeg installation
   - Verify audio file format

3. **Out of memory**
   - Reduce batch_size
   - Use batch processor instead of parallel

4. **Slow processing**
   - Use parallel processor
   - Increase max_workers
   - Check internet connection (API calls)

---

## Future Enhancements

1. **Streaming responses** - Real-time answer generation
2. **Multi-language support** - Non-English documents
3. **Advanced retrieval** - Hybrid search, reranking
4. **Caching** - Reduce API calls
5. **Async processing** - Non-blocking operations
6. **Web scraping** - URL ingestion
7. **OCR support** - Image-based PDFs

---

## License & Credits

- **LangChain** - RAG framework
- **OpenAI** - LLM and embeddings
- **Whisper** - Audio transcription
- **Chroma** - Vector database
- **Gradio** - Web UI framework

---

**Last Updated:** January 2025
**Version:** 1.0.0
