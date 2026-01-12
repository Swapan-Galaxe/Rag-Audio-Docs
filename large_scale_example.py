"""
Complete example for processing large datasets
Choose the approach based on your needs
"""

# Approach 1: BATCH PROCESSING (Sequential, Memory-efficient)
def batch_approach():
    from batch_processor import BatchRAGProcessor
    
    processor = BatchRAGProcessor(persist_directory="./batch_vectordb")
    
    # Load and process
    docs = processor.load_directory("./data")
    chunks = processor.process_in_batches(docs, batch_size=100)
    processor.create_vectorstore(chunks, batch_size=500)
    
    # Query
    answer = processor.query("Summarize the main topics")
    print(answer)
    print(processor.get_stats())


# Approach 2: PARALLEL PROCESSING (Fast, CPU-intensive)
def parallel_approach():
    from parallel_processor import ParallelRAGProcessor
    
    processor = ParallelRAGProcessor(
        persist_directory="./parallel_vectordb",
        max_workers=4
    )
    
    # Process everything in parallel
    chunks = processor.process_documents("./data", batch_size=500)
    
    # Query
    answer = processor.query("What are the key findings?")
    print(answer)


# Approach 3: INCREMENTAL PROCESSING (Smart, Only new files)
def incremental_approach():
    from incremental_processor import IncrementalProcessor
    
    processor = IncrementalProcessor(persist_directory="./incremental_vectordb")
    
    # First run: processes all files
    processor.process_new_files("./data", batch_size=100)
    
    # Add more files to ./data directory
    # Second run: only processes new files
    processor.process_new_files("./data", batch_size=100)
    
    # Query
    answer = processor.query("What's new?")
    print(answer)


# RECOMMENDED: Start with this
def recommended_approach():
    """
    Use incremental + parallel for best results
    """
    from incremental_processor import IncrementalProcessor
    
    processor = IncrementalProcessor(persist_directory="./my_vectordb")
    
    # Process new files only
    processor.process_new_files("./data", batch_size=100)
    
    # Query with more context
    answer = processor.query("Provide a comprehensive summary", k=10)
    print(answer)
    
    print(f"\n{processor.get_stats()}")


if __name__ == "__main__":
    # Choose one:
    # batch_approach()
    # parallel_approach()
    # incremental_approach()
    recommended_approach()
