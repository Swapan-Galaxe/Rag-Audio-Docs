# RAG Pipeline for Audio & Document Summarization

Extract summaries from audio files and documents using LangChain RAG pipeline.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file with your API keys:
```
OPENAI_API_KEY=your_key_here

# Optional: Enable LangSmith tracing
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=rag-audio-docs
```

Get your LangSmith API key from: https://smith.langchain.com/

## Quick Start

```python
from rag_summarizer import RAGSummarizer

summarizer = RAGSummarizer()

# Process files
chunks = summarizer.process_files(
    document_paths=["document.pdf"],
    audio_paths=["audio.mp3"]
)

# Get summary
summary = summarizer.generate_summary(chunks)
print(summary)

# Query
answer = summarizer.query("What are the main points?")
print(answer)
```

## Usage

### Basic Summarization
```python
summarizer = RAGSummarizer()
chunks = summarizer.process_files(document_paths=["file.pdf"])
summary = summarizer.generate_summary(chunks, chain_type="map_reduce")
```

### Custom Summary Query
```python
summary = summarizer.custom_summary("Summarize the key findings")
```

### Q&A on Documents
```python
answer = summarizer.query("What is the conclusion?")
```

## Supported Formats

- **Documents**: PDF, DOCX, TXT
- **Audio**: MP3, WAV, M4A (any format supported by Whisper)

## Chain Types

- `map_reduce`: Parallel processing (faster, good for long docs)
- `refine`: Sequential refinement (more coherent)
- `stuff`: Single prompt (best for short docs)
