import os
from rag_summarizer import RAGSummarizer
from langchain_core.documents import Document

def test_with_sample_data():
    """Test RAG pipeline with generated sample data (no files needed)"""
    
    print("Testing RAG Summarizer with sample data...\n")
    
    # Create sample documents directly
    sample_docs = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that enables systems to learn from data. It includes supervised learning, unsupervised learning, and reinforcement learning.",
            metadata={"source": "sample_doc1.txt"}
        ),
        Document(
            page_content="Deep learning uses neural networks with multiple layers. It has revolutionized computer vision, natural language processing, and speech recognition.",
            metadata={"source": "sample_doc2.txt"}
        ),
        Document(
            page_content="Natural language processing allows computers to understand human language. Applications include chatbots, translation, and sentiment analysis.",
            metadata={"source": "sample_doc3.txt"}
        )
    ]
    
    # Initialize summarizer
    summarizer = RAGSummarizer()
    
    # Process documents
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(sample_docs)
    
    # Create vector store
    from langchain_community.vectorstores import Chroma
    summarizer.vectorstore = Chroma.from_documents(chunks, summarizer.embeddings)
    
    print(f"Processed {len(chunks)} chunks\n")
    
    # Test 1: Generate summary
    print("=" * 50)
    print("Test 1: Generate Summary")
    print("=" * 50)
    summary = summarizer.generate_summary(chunks, chain_type="stuff")
    print(f"\nSummary:\n{summary}\n")
    
    # Test 2: Query
    print("=" * 50)
    print("Test 2: Query RAG System")
    print("=" * 50)
    question = "What is deep learning?"
    answer = summarizer.query(question)
    print(f"\nQ: {question}")
    print(f"A: {answer}\n")
    
    # Test 3: Custom summary
    print("=" * 50)
    print("Test 3: Custom Summary")
    print("=" * 50)
    custom = summarizer.custom_summary("List the main AI topics mentioned")
    print(f"\n{custom}\n")
    
    print("All tests passed!")

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY in .env file")
        print("\nCreate .env file with:")
        print("OPENAI_API_KEY=your_key_here")
    else:
        test_with_sample_data()
