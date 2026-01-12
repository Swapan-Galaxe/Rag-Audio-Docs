from batch_processor import BatchRAGProcessor

# Initialize
processor = BatchRAGProcessor(persist_directory="./test_vectordb")

# Load documents
docs = processor.load_directory("./data/documents", file_types=["*.txt", "*.pdf"])
print(f"Loaded {len(docs)} documents")

# Load audio files
audio_docs = processor.process_audio_batch("./data/audio")
print(f"Transcribed {len(audio_docs)} audio files\n")

# Combine all documents
all_docs = docs + audio_docs
print(f"Total documents: {len(all_docs)}\n")

# Process in batches
chunks = processor.process_in_batches(all_docs, batch_size=50)
print(f"Created {len(chunks)} chunks\n")

# Create vector store
processor.create_vectorstore(chunks, batch_size=100)

# Generate summary using retrieval (FAST - works with all docs)
print("=" * 60)
print("GENERATING SUMMARY...")
print("=" * 60)
summary = processor.custom_summary("Provide a comprehensive summary of all the content")
print(summary)
print("=" * 60 + "\n")

# Test queries
questions = [
    "What is machine learning?",
    "What are cloud service providers?",
    "What was discussed in the audio?"
]

print("\n" + "=" * 60)
print("Q&A SESSION")
print("=" * 60)
for q in questions:
    print(f"Q: {q}")
    answer = processor.query(q, k=3)
    print(f"A: {answer}\n")

# Stats
print("=" * 60)
print(processor.get_stats())
print("=" * 60)
