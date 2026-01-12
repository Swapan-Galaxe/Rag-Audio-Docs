import os
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential
import whisper
from dotenv import load_dotenv
from tqdm import tqdm
import glob

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

# Add ffmpeg to PATH
ffmpeg_path = Path(__file__).parent / "ffmpeg-master-latest-win64-gpl" / "bin"
if str(ffmpeg_path) not in os.environ["PATH"]:
    os.environ["PATH"] = str(ffmpeg_path) + os.pathsep + os.environ["PATH"]

class BatchRAGProcessor:
    def __init__(self, model_name="gpt-3.5-turbo", persist_directory="./chroma_db"):
        self.llm = ChatOpenAI(temperature=0, model=model_name)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.persist_directory = persist_directory
        logger.info(f"Initialized BatchRAGProcessor with model: {model_name}")
        
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
        
    def load_directory(self, directory_path, file_types=["*.pdf", "*.txt", "*.docx"]):
        """Load all files from directory with error handling"""
        logger.info(f"Loading files from {directory_path}...")
        all_docs = []
        
        for file_type in file_types:
            files = glob.glob(os.path.join(directory_path, "**", file_type), recursive=True)
            logger.info(f"Found {len(files)} {file_type} files")
            
            for file_path in tqdm(files, desc=f"Loading {file_type}"):
                try:
                    if file_path.endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                    elif file_path.endswith('.docx'):
                        loader = Docx2txtLoader(file_path)
                    else:
                        loader = TextLoader(file_path, encoding='utf-8')
                    all_docs.extend(loader.load())
                    logger.debug(f"Successfully loaded {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    continue
        
        logger.info(f"Total documents loaded: {len(all_docs)}")
        return all_docs
    
    def process_in_batches(self, docs, batch_size=100):
        """Process documents in batches to avoid memory issues"""
        logger.info(f"Processing {len(docs)} documents in batches of {batch_size}...")
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_chunks = []
        
        for i in tqdm(range(0, len(docs), batch_size), desc="Splitting batches"):
            batch = docs[i:i + batch_size]
            chunks = splitter.split_documents(batch)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks")
        return all_chunks
    
    def create_vectorstore(self, chunks, batch_size=500):
        """Create persistent vector store with batching for large datasets"""
        logger.info(f"Creating vector store with {len(chunks)} chunks...")
        
        if os.path.exists(self.persist_directory):
            logger.info("Loading existing vector store...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            logger.info("Adding new documents...")
            for i in tqdm(range(0, len(chunks), batch_size), desc="Adding batches"):
                batch = chunks[i:i + batch_size]
                self.vectorstore.add_documents(batch)
        else:
            logger.info("Creating new vector store...")
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
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def process_audio_batch(self, audio_directory):
        """Process multiple audio files with retry logic"""
        audio_files = glob.glob(os.path.join(audio_directory, "**/*.mp3"), recursive=True)
        audio_files.extend(glob.glob(os.path.join(audio_directory, "**/*.wav"), recursive=True))
        
        logger.info(f"Found {len(audio_files)} audio files")
        model = whisper.load_model("base")
        
        docs = []
        for audio_path in tqdm(audio_files, desc="Transcribing audio"):
            try:
                logger.info(f"Transcribing {audio_path}...")
                result = model.transcribe(audio_path)
                docs.append(Document(
                    page_content=result["text"],
                    metadata={"source": audio_path, "type": "audio"}
                ))
                logger.debug(f"Successfully transcribed {audio_path}")
            except Exception as e:
                logger.error(f"Error transcribing {audio_path}: {e}")
                continue
        
        logger.info(f"Transcribed {len(docs)} audio files")
        return docs
    
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
    
    # def generate_summary(self, docs, max_tokens=500):
    #     """Generate comprehensive summary from all documents using map_reduce"""
    #     from langchain_classic.chains.summarize import load_summarize_chain
    #     
    #     print(f"Generating summary from {len(docs)} documents...")
    #     print("This may take several minutes for large document sets...")
    #     chain = load_summarize_chain(self.llm, chain_type="map_reduce", verbose=True)
    #     summary = chain.run(docs)
    #     return summary
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def custom_summary(self, query="Provide a comprehensive summary"):
        """Generate custom summary with MMR retrieval and retry logic"""
        if not self.vectorstore:
            raise ValueError("No vector store created. Call create_vectorstore first.")
        
        try:
            logger.info(f"Generating custom summary with query: {query}")
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 10, "fetch_k": 20}
            )
            docs = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            prompt = ChatPromptTemplate.from_template(
                "Based on the following context, {query}:\n\n{context}"
            )
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({"query": query, "context": context})
            logger.info("Custom summary generated successfully")
            return result
        except Exception as e:
            logger.error(f"Error generating custom summary: {e}")
            raise
    
    def get_stats(self):
        """Get statistics about the vector store"""
        if self.vectorstore:
            count = self.vectorstore._collection.count()
            logger.info(f"Vector store contains {count} chunks")
            return f"Vector store contains {count} chunks"
        return "No vector store created"
