# Testing Guide

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file:
```
OPENAI_API_KEY=your_actual_key_here
```

## Test Options

### Option 1: Quick Test (Easiest)
Uses included sample text files:
```bash
python quick_test.py
```

### Option 2: Test with Sample Data (No files needed)
Generates test data programmatically:
```bash
python test_sample.py
```

### Option 3: Test with Your Files
Edit `test_files.py` and add your file paths:
```python
document_paths = ["your_file.pdf", "your_doc.txt"]
audio_paths = ["your_audio.mp3"]
```
Then run:
```bash
python test_files.py
```

## Expected Output

All tests should show:
- ✓ Number of chunks processed
- Summary of the content
- Answers to queries
- No errors

## Troubleshooting

**Error: OPENAI_API_KEY not found**
- Create `.env` file with your API key

**Error: File not found**
- Check file paths are correct
- Use absolute paths if needed

**Error: Module not found**
- Run: `pip install -r requirements.txt`

**Whisper audio errors**
- Ensure ffmpeg is installed for audio processing
- Windows: `choco install ffmpeg`
- Or use online transcription API instead

## Test Individual Components

```python
from rag_summarizer import RAGSummarizer

# Test document loading only
summarizer = RAGSummarizer()
docs = summarizer.load_documents(["sample_doc.txt"])
print(f"Loaded {len(docs)} documents")

# Test audio transcription only
text = summarizer.transcribe_audio("audio.mp3")
print(text)
```
