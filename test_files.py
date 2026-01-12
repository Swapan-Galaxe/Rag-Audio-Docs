import os
from rag_summarizer import RAGSummarizer

def test_with_files():
    """Test with actual PDF, TXT, or audio files"""
    
    # REPLACE THESE WITH YOUR ACTUAL FILE PATHS
    document_paths = [
        "C:/Users/sdeep/Downloads/Deep+Learning+Ian+Goodfellow.pdf",      # Replace with your PDF
        "C:/Users/sdeep/Downloads/In machine learning, deep learning.txt",    # Replace with your TXT file
    ]
    
    audio_paths = [
         "C:\\Users\\sdeep\\Downloads\\resources_RC_Conversation_Sample.mp3",
    ]
    
    print("Testing RAG Summarizer with actual files...\n")
    
    # Initialize
    summarizer = RAGSummarizer()
    
    # Process files
    try:
        chunks = summarizer.process_files(
            document_paths=document_paths,
            audio_paths=audio_paths if audio_paths else None
        )
        
        print(f"\nSuccessfully processed {len(chunks)} chunks\n")
        
        # Generate summary
        print("=" * 50)
        print("Generating Summary (first 10 chunks)...")
        print("=" * 50)
        summary = summarizer.generate_summary(chunks)
        print(f"\n{summary}\n")
        
        # Generate comprehensive summary from all files
        print("=" * 50)
        print("Generating Comprehensive Summary (all files)...")
        print("=" * 50)
        comprehensive = summarizer.custom_summary("Provide a comprehensive summary covering all topics from all documents and audio")
        print(f"\n{comprehensive}\n")
        
        # Query
        print("=" * 50)
        print("Querying Documents...")
        print("=" * 50)
        answer = summarizer.query("What are the key points?")
        print(f"\n{answer}\n")
        
        print("Test completed successfully!")
        
    except FileNotFoundError as e:
        print(f"ERROR: File not found - {e}")
        print("\nUpdate file paths in test_files.py with your actual files")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY in .env file")
    else:
        test_with_files()
