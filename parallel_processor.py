import os
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from tqdm import tqdm
import glob
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable LangSmith tracing
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "rag-audio-docs")
    logger.info(f"LangSmith tracing enabled - Project: {os.environ['LANGCHAIN_PROJECT']}")
else:
    logger.info("LangSmith tracing disabled (no API key found)")

class ParallelRAGProcessor:
    def __init__(self, model_name="gpt-3.5-turbo", persist_directory="./chroma_db", max_workers=4):
        self.llm = ChatOpenAI(temperature=0, model=model_name)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.persist_directory = persist_directory
        self.max_workers = max_workers
        logger.info(f"Initialized ParallelRAGProcessor with {max_workers} workers")
        
        # Load existing vector store if available
        if Path(persist_directory).exists():
            try:
                self.vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info(f"Loaded existing vector store from {persist_directory}")
            except Exception as e:
                logger.warning(f"Could not load existing vector store: {e}")
    
    def load_single_file(self, file_path):
        """Load a single file with error handling"""
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding='utf-8')
            logger.debug(f"Successfully loaded {file_path}")
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []
    
    def load_directory_parallel(self, directory_path, file_types=["*.pdf", "*.txt", "*.docx"]):
        """Load files in parallel"""
        logger.info(f"Loading files from {directory_path}...")
        
        all_files = []
        for file_type in file_types:
            files = glob.glob(os.path.join(directory_path, "**", file_type), recursive=True)
            all_files.extend(files)
        
        logger.info(f"Found {len(all_files)} files")
        
        all_docs = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(tqdm(
                executor.map(self.load_single_file, all_files),
                total=len(all_files),
                desc="Loading files"
            ))
        
        for docs in results:
            all_docs.extend(docs)
        
        logger.info(f"Loaded {len(all_docs)} documents")
        return all_docs
    
    def create_vectorstore(self, chunks, batch_size=500):
        """Create persistent vector store with batching"""
        logger.info(f"Creating vector store with {len(chunks)} chunks...")
        
        if os.path.exists(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            for i in tqdm(range(0, len(chunks), batch_size), desc="Adding batches"):
                batch = chunks[i:i + batch_size]
                self.vectorstore.add_documents(batch)
        else:
            self.vectorstore = Chroma.from_documents(
                chunks[:batch_size],
                self.embeddings,
                persist_directory=self.persist_directory
            )
            for i in tqdm(range(batch_size, len(chunks), batch_size), desc="Adding batches"):
                batch = chunks[i:i + batch_size]
                self.vectorstore.add_documents(batch)
        
        logger.info(f"Vector store created at {self.persist_directory}")
        return self.vectorstore
    
    def process_documents(self, directory_path, batch_size=500):
        """Complete pipeline"""
        docs = self.load_directory_parallel(directory_path)
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        logger.info(f"Created {len(chunks)} chunks")
        
        self.create_vectorstore(chunks, batch_size)
        return chunks
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def query(self, question, k=5):
        """Query the vector store with MMR retrieval and retry logic"""
        if not self.vectorstore:
            raise ValueError("No vector store created")
        
        try:
            logger.info(f"Processing query: {question}")
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": k, "fetch_k": k * 2}
                ),
                chain_type="stuff"
            )
            result = qa_chain.run(question)
            logger.info("Query processed successfully")
            return result
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
