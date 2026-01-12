from batch_processor import BatchRAGProcessor
import os
import json
import logging
from datetime import datetime
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import Chroma
import glob

logger = logging.getLogger(__name__)

class IncrementalProcessor(BatchRAGProcessor):
    def __init__(self, model_name="gpt-3.5-turbo", persist_directory="./chroma_db"):
        super().__init__(model_name, persist_directory)
        self.processed_files_log = os.path.join(persist_directory, "processed_files.json")
        self.processed_files = self.load_processed_files()
        logger.info(f"Initialized IncrementalProcessor with {len(self.processed_files)} previously processed files")
    
    def load_processed_files(self):
        """Load list of already processed files"""
        if os.path.exists(self.processed_files_log):
            try:
                with open(self.processed_files_log, 'r') as f:
                    files = json.load(f)
                logger.info(f"Loaded {len(files)} processed files from log")
                return files
            except Exception as e:
                logger.error(f"Error loading processed files log: {e}")
                return {}
        return {}
    
    def save_processed_files(self):
        """Save list of processed files"""
        try:
            os.makedirs(os.path.dirname(self.processed_files_log), exist_ok=True)
            with open(self.processed_files_log, 'w') as f:
                json.dump(self.processed_files, f, indent=2)
            logger.info(f"Saved {len(self.processed_files)} processed files to log")
        except Exception as e:
            logger.error(f"Error saving processed files log: {e}")
    
    def get_new_files(self, directory_path, file_types=["*.pdf", "*.txt", "*.docx"]):
        """Get only new files that haven't been processed"""
        all_files = []
        for file_type in file_types:
            files = glob.glob(os.path.join(directory_path, "**", file_type), recursive=True)
            all_files.extend(files)
        
        new_files = [f for f in all_files if f not in self.processed_files]
        logger.info(f"Found {len(new_files)} new files out of {len(all_files)} total")
        return new_files
    
    def process_new_files(self, directory_path, batch_size=100):
        """Process only new files"""
        new_files = self.get_new_files(directory_path)
        
        if not new_files:
            logger.info("No new files to process")
            return
        
        logger.info(f"Processing {len(new_files)} new files...")
        all_docs = []
        
        for file_path in tqdm(new_files, desc="Loading new files"):
            try:
                if file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                elif file_path.endswith('.docx'):
                    loader = Docx2txtLoader(file_path)
                else:
                    loader = TextLoader(file_path, encoding='utf-8')
                
                docs = loader.load()
                all_docs.extend(docs)
                self.processed_files[file_path] = datetime.now().isoformat()
                logger.debug(f"Successfully loaded {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        if not all_docs:
            logger.warning("No documents were successfully loaded")
            return
        
        # Process and add to existing vector store
        chunks = self.process_in_batches(all_docs, batch_size)
        
        if os.path.exists(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            logger.info("Adding to existing vector store...")
            self.vectorstore.add_documents(chunks)
        else:
            self.create_vectorstore(chunks)
        
        self.save_processed_files()
        logger.info(f"Successfully processed {len(new_files)} new files")
