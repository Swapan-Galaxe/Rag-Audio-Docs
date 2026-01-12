import os
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential
import whisper
from dotenv import load_dotenv

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

class RAGSummarizer:
    def __init__(self, model_name="gpt-3.5-turbo", persist_directory="./chroma_db"):
        self.llm = ChatOpenAI(temperature=0, model=model_name)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.persist_directory = persist_directory
        logger.info(f"Initialized RAGSummarizer with model: {model_name}")
        
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
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def transcribe_audio(self, audio_path):
        """Transcribe audio file to text using Whisper with retry"""
        try:
            logger.info(f"Transcribing {audio_path}...")
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            logger.info(f"Successfully transcribed {audio_path}")
            return result["text"]
        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {e}")
            raise
    
    def load_documents(self, file_paths):
        """Load documents from various file formats with error handling"""
        docs = []
        for path in file_paths:
            try:
                logger.info(f"Loading {path}...")
                if path.endswith('.pdf'):
                    loader = PyPDFLoader(path)
                elif path.endswith('.docx'):
                    loader = Docx2txtLoader(path)
                else:
                    loader = TextLoader(path, encoding='utf-8')
                docs.extend(loader.load())
                logger.info(f"Successfully loaded {path}")
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")
                continue
        return docs
    
    def process_files(self, document_paths=None, audio_paths=None):
        """Process documents and audio files into persistent vector store"""
        all_docs = []
        
        try:
            if document_paths:
                all_docs.extend(self.load_documents(document_paths))
            
            if audio_paths:
                for audio in audio_paths:
                    text = self.transcribe_audio(audio)
                    all_docs.append(Document(
                        page_content=text,
                        metadata={"source": audio, "type": "audio"}
                    ))
            
            if not all_docs:
                raise ValueError("No documents were successfully loaded")
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(all_docs)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Create or update persistent vector store
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    chunks,
                    self.embeddings,
                    persist_directory=self.persist_directory
                )
                logger.info(f"Created new vector store at {self.persist_directory}")
            else:
                self.vectorstore.add_documents(chunks)
                logger.info(f"Added documents to existing vector store")
            
            return chunks
        except Exception as e:
            logger.error(f"Error processing files: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_summary(self, chunks, chain_type="map_reduce"):
        """Generate summary with retry logic"""
        try:
            logger.info(f"Generating summary using {chain_type}...")
            combined_text = "\n\n".join([doc.page_content for doc in chunks[:10]])
            prompt = ChatPromptTemplate.from_template(
                "Provide a comprehensive summary of the following text:\n\n{text}"
            )
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({"text": combined_text})
            logger.info("Summary generated successfully")
            return result
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def query(self, question):
        """Query the RAG system with MMR retrieval and retry logic"""
        if not self.vectorstore:
            raise ValueError("No documents processed. Call process_files first.")
        
        try:
            logger.info(f"Processing query: {question}")
            # Use MMR for diverse results
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3, "fetch_k": 10}
            )
            docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            prompt = ChatPromptTemplate.from_template(
                "Answer the question based on the following context:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
            )
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({"context": context, "question": question})
            logger.info("Query processed successfully")
            return result
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def custom_summary(self, query="Provide a comprehensive summary"):
        """Generate custom summary with MMR retrieval and retry logic"""
        if not self.vectorstore:
            raise ValueError("No documents processed. Call process_files first.")
        
        try:
            logger.info(f"Generating custom summary with query: {query}")
            prompt = ChatPromptTemplate.from_template(
                "Based on the following context, {query}:\n\n{context}"
            )
            
            # Use MMR for diverse results
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 20}
            )
            docs = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({"query": query, "context": context})
            logger.info("Custom summary generated successfully")
            return result
        except Exception as e:
            logger.error(f"Error generating custom summary: {e}")
            raise
