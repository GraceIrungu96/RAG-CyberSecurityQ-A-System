"""
Cybersecurity Awareness Assistant - RAG System
A comprehensive system that helps users understand cybersecurity threats using RAG architecture
"""

import os
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import asyncio
import json
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import PyPDF2
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from groq import Groq
import chromadb
from chromadb.config import Settings
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re

# Download required NLTK data with proper error handling
def download_nltk_data():
    """Download required NLTK data with fallback options"""
    required_data = [
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords')
    ]
    
    for resource_path, resource_name in required_data:
        try:
            nltk.data.find(resource_path)
            print(f"✓ {resource_name} already available")
        except LookupError:
            try:
                print(f"Downloading {resource_name}...")
                nltk.download(resource_name, quiet=True)
                print(f"✓ {resource_name} downloaded successfully")
            except Exception as e:
                print(f"⚠ Warning: Could not download {resource_name}: {e}")

# Initialize NLTK data
download_nltk_data()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GROQ_API_KEY = "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"
MODEL_NAME = "gemma2-9b-it"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

@dataclass
class CyberSecuritySource:
    """Data class for cybersecurity knowledge sources"""
    name: str
    type: str  # 'pdf', 'url', 'text'
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class DocumentProcessor:
    """Handles processing of various document types"""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            # Fallback to basic English stopwords if NLTK corpus not available
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once'
            }
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF files"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks with fallback sentence splitting"""
        try:
            # Try using NLTK sentence tokenizer
            sentences = sent_tokenize(text)
        except Exception as e:
            logger.warning(f"NLTK sentence tokenizer failed, using simple splitting: {e}")
            # Fallback to simple sentence splitting
            sentences = self._simple_sentence_split(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep overlap
                overlap_sentences = current_chunk[-overlap//10:] if len(current_chunk) > overlap//10 else current_chunk
                current_chunk = overlap_sentences
                current_length = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting fallback"""
        # Split on sentence ending punctuation
        sentences = re.split(r'[.!?]+', text)
        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove special characters and normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower().strip()
        return text

class VectorStore:
    """Vector database for storing and retrieving document embeddings"""
    
    def __init__(self, collection_name: str = "cybersecurity_knowledge"):
        try:
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./chroma_db"
            ))
        except Exception as e:
            logger.warning(f"ChromaDB initialization failed, using in-memory client: {e}")
            self.client = chromadb.Client()
        
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.is_fitted = False
    
    def add_documents(self, sources: List[CyberSecuritySource]):
        """Add documents to the vector store"""
        documents = []
        metadatas = []
        ids = []
        
        doc_processor = DocumentProcessor()
        
        for i, source in enumerate(sources):
            chunks = doc_processor.chunk_text(source.content)
            for j, chunk in enumerate(chunks):
                doc_id = f"{source.name}_{i}_{j}_{datetime.now().timestamp()}"
                documents.append(chunk)
                metadatas.append({
                    "source_name": source.name,
                    "source_type": source.type,
                    "chunk_id": j,
                    **source.metadata
                })
                ids.append(doc_id)
        
        try:
            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            # Fit TF-IDF vectorizer
            if not self.is_fitted:
                self.vectorizer.fit(documents)
                self.is_fitted = True
            
            logger.info(f"Added {len(documents)} document chunks to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        try:
            # Search using ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )
            
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    search_results.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results['distances'] else 0
                    })
            
            return search_results
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []

class GroqLLM:
    """Groq API wrapper for text generation"""
    
    def __init__(self, api_key: str, model: str = MODEL_NAME):
        self.client = Groq(api_key=api_key)
        self.model = model
    
    def generate_response(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate response using Groq API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a cybersecurity expert assistant. Provide clear, accurate, and actionable advice to help users understand and protect against cyber threats. Use simple language that non-technical users can understand."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."

class CybersecurityRAGSystem:
    """Main RAG system for cybersecurity assistance"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm = GroqLLM(GROQ_API_KEY)
        self.document_processor = DocumentProcessor()
        self.knowledge_base = []
        
        # Initialize with default cybersecurity knowledge
        self._initialize_default_knowledge()
    
    def _initialize_default_knowledge(self):
        """Initialize with basic cybersecurity knowledge"""
        default_sources = [
            CyberSecuritySource(
                name="Phishing Basics",
                type="text",
                content="""
                Phishing is a cybercrime where attackers impersonate legitimate organizations to steal sensitive information like passwords, credit card numbers, or personal data. Common signs of phishing emails include: urgent language demanding immediate action, suspicious sender addresses that don't match the claimed organization, generic greetings like 'Dear Customer', requests for personal information, suspicious links or attachments, poor grammar and spelling errors. To protect yourself: verify sender identity independently, don't click suspicious links, check URLs carefully, use multi-factor authentication, keep software updated, and report suspicious emails to your IT department or email provider.
                """,
                metadata={"topic": "phishing", "difficulty": "beginner"}
            ),
            CyberSecuritySource(
                name="Password Security",
                type="text",
                content="""
                Strong passwords are your first line of defense against cyber attacks. Weak passwords can be easily cracked by attackers using automated tools. Create strong passwords by using at least 12 characters, mixing uppercase and lowercase letters, numbers, and special characters. Avoid common words, personal information, or predictable patterns. Use unique passwords for each account. Consider using passphrases - longer phrases that are easier to remember but hard to crack. Enable multi-factor authentication whenever possible. Use a reputable password manager to generate and store complex passwords securely. Change passwords immediately if you suspect a breach.
                """,
                metadata={"topic": "passwords", "difficulty": "beginner"}
            ),
            CyberSecuritySource(
                name="Malware Protection",
                type="text",
                content="""
                Malware includes viruses, trojans, ransomware, spyware, and other malicious software designed to harm your devices or steal information. Common infection methods include email attachments, malicious downloads, infected USB drives, and compromised websites. Protect yourself by keeping operating systems and software updated with latest security patches, using reputable antivirus software with real-time protection, being cautious with email attachments and downloads, avoiding suspicious websites and pop-up ads, regularly backing up important data, and using standard user accounts instead of administrator accounts for daily activities.
                """,
                metadata={"topic": "malware", "difficulty": "intermediate"}
            ),
            CyberSecuritySource(
                name="Social Engineering",
                type="text",
                content="""
                Social engineering attacks manipulate human psychology to trick people into revealing confidential information or performing actions that compromise security. Common techniques include pretexting (creating fake scenarios), baiting (offering something enticing), tailgating (following someone into secure areas), and quid pro quo (offering services in exchange for information). Protect yourself by being skeptical of unsolicited communications, verifying identities independently, limiting information shared on social media, following company security policies, and trusting your instincts when something seems off.
                """,
                metadata={"topic": "social_engineering", "difficulty": "intermediate"}
            )
        ]
        
        self.vector_store.add_documents(default_sources)
        logger.info("Initialized default cybersecurity knowledge base")
    
    def add_pdf_document(self, pdf_path: str, source_name: str, metadata: Dict[str, Any] = None):
        """Add PDF document to knowledge base"""
        if metadata is None:
            metadata = {}
        
        text = self.document_processor.extract_pdf_text(pdf_path)
        if text:
            source = CyberSecuritySource(
                name=source_name,
                type="pdf",
                content=text,
                metadata={**metadata, "file_path": pdf_path}
            )
            self.vector_store.add_documents([source])
            logger.info(f"Added PDF document: {source_name}")
        else:
            logger.error(f"Failed to extract text from PDF: {pdf_path}")
    
    def query(self, user_question: str, context_limit: int = 3) -> Dict[str, Any]:
        """Process user query and generate response"""
        # Retrieve relevant context
        relevant_docs = self.vector_store.search(user_question, k=context_limit)
        
        # Build context from retrieved documents
        context = "\n\n".join([doc['content'] for doc in relevant_docs])
        
        # Create prompt for LLM
        prompt = f"""
        Based on the following cybersecurity knowledge and the user's question, provide a helpful, accurate, and easy-to-understand response.

        Context from cybersecurity knowledge base:
        {context}

        User Question: {user_question}

        Please provide a comprehensive answer that:
        1. Directly addresses the user's question
        2. Uses simple, non-technical language
        3. Provides actionable advice
        4. Includes specific examples when helpful
        5. Mentions potential risks and how to mitigate them

        Answer:
        """
        
        # Generate response
        response = self.llm.generate_response(prompt)
        
        return {
            "answer": response,
            "sources": [doc['metadata'] for doc in relevant_docs],
            "context_used": len(relevant_docs),
            "timestamp": datetime.now().isoformat()
        }

# FastAPI Application
app = FastAPI(
    title="Cybersecurity Awareness Assistant",
    description="AI-powered assistant to help users understand and protect against cyber threats",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
rag_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    global rag_system
    try:
        rag_system = CybersecurityRAGSystem()
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise

# Pydantic models for API
class QueryRequest(BaseModel):
    question: str
    context_limit: Optional[int] = 3

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    context_used: int
    timestamp: str

@app.post("/query", response_model=QueryResponse)
async def query_assistant(request: QueryRequest):
    """Main endpoint for asking cybersecurity questions"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        result = rag_system.query(request.question, request.context_limit)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Error processing your question")

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    source_name: str = Form(...),
    topic: str = Form(None),
    difficulty: str = Form("intermediate")
):
    """Upload PDF documents to expand knowledge base"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Add to knowledge base
        metadata = {
            "topic": topic or "general",
            "difficulty": difficulty,
            "uploaded_filename": file.filename
        }
        
        rag_system.add_pdf_document(temp_path, source_name, metadata)
        
        # Clean up temporary file
        try:
            os.remove(temp_path)
        except Exception as cleanup_error:
            logger.warning(f"Could not remove temporary file: {cleanup_error}")
        
        return {"message": f"Successfully added PDF: {source_name}"}
    
    except Exception as e:
        logger.error(f"Error uploading PDF: {e}")
        raise HTTPException(status_code=500, detail="Error processing PDF upload")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if rag_system else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "rag_system_initialized": rag_system is not None
    }

@app.get("/topics")
async def get_topics():
    """Get available cybersecurity topics"""
    return {
        "topics": [
            "phishing",
            "passwords",
            "malware",
            "social_engineering",
            "network_security",
            "data_protection",
            "incident_response",
            "security_awareness"
        ]
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Cybersecurity Awareness Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "query": "POST /query - Ask cybersecurity questions",
            "upload": "POST /upload-pdf - Upload PDF documents",
            "health": "GET /health - Health check",
            "topics": "GET /topics - Available topics"
        }
    }

def test_system():
    """Test the RAG system functionality"""
    print("Starting Cybersecurity Awareness Assistant...")
    
    try:
        # Initialize the system
        global rag_system
        rag_system = CybersecurityRAGSystem()
        
        # Test the system
        test_questions = [
            "How do I know if this email is a phishing attempt?",
            "What makes a password strong?",
            "How can I protect my computer from malware?",
            "What is social engineering and how do I protect myself?"
        ]
        
        print("\nTesting the RAG system:")
        for question in test_questions:
            print(f"\nQ: {question}")
            try:
                result = rag_system.query(question)
                print(f"A: {result['answer'][:200]}...")
            except Exception as e:
                print(f"Error processing question: {e}")
        
        print("\n✓ RAG system test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ RAG system test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the system first
    if test_system():
        # Start the FastAPI server
        print("\nStarting FastAPI server...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("System test failed. Please check the configuration and try again.")