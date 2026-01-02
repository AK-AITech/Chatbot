"""
LangChain + Gemini API Chatbot Backend
FastAPI server with LangChain integration and document processing
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import sqlite3
import uuid
from datetime import datetime
from typing import List, Optional
import aiofiles
import PyPDF2
import io

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LangChain LLM with Gemini
# We will now initialize this dynamically per request
# llm = ChatGoogleGenerativeAI(...)

# Create a chat prompt template with optimized system instructions
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a precise and helpful AI assistant.

Guidelines:
- Provide accurate, concise answers
- Ask for clarification if unclear
- Reference documents when relevant
- Stay focused and professional
- Give direct, actionable responses"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Helper to get LLM based on provider
def get_llm(provider: str, model: str = None):
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY not found in environment variables")
        
        return ChatOpenAI(
            model=model or "gpt-4o",
            api_key=api_key,
            temperature=0.2
        )
    else:
        # Default to Gemini
        return ChatGoogleGenerativeAI(
            model=model or "gemini-2.0-flash-exp",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2,
            max_output_tokens=4096,
            convert_system_message_to_human=True
        )


# Initialize database


def init_db():
    conn = sqlite3.connect('chatbot_memory.db')
    cursor = conn.cursor()

    # Create conversations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            message_type TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create documents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT,
            content TEXT,
            file_type TEXT,
            upload_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()


# Initialize database on startup
init_db()

# Global variables for conversation management
session_histories = {}  # Store chat histories per session
stored_documents = []

# Request models


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    include_documents: bool = True
    provider: str = "gemini"
    model: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    message: str
    document_id: str
    filename: str

# Memory management functions


def get_or_create_history(session_id: str):
    """Get or create chat message history for a session"""
    if session_id not in session_histories:
        session_histories[session_id] = InMemoryChatMessageHistory()
    return session_histories[session_id]


def save_to_db(session_id: str, message_type: str, content: str):
    """Save message to database"""
    conn = sqlite3.connect('chatbot_memory.db')
    cursor = conn.cursor()

    message_id = str(uuid.uuid4())
    cursor.execute('''
        INSERT INTO conversations (id, session_id, message_type, content)
        VALUES (?, ?, ?, ?)
    ''', (message_id, session_id, message_type, content))

    conn.commit()
    conn.close()


def load_conversation_history(session_id: str):
    """Load conversation history from database"""
    conn = sqlite3.connect('chatbot_memory.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT message_type, content, timestamp
        FROM conversations
        WHERE session_id = ?
        ORDER BY timestamp ASC
    ''', (session_id,))

    history = cursor.fetchall()
    conn.close()
    return history

# Document processing functions


async def process_document(file: UploadFile) -> str:
    """Process uploaded document and return content"""
    content = ""

    # Read file content
    file_content = await file.read()

    try:
        # Process based on file type
        if file.filename.endswith('.pdf'):
            # Process PDF using PyPDF2
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            content = ""
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"

        elif file.filename.endswith(('.txt', '.md')):
            # Process text files
            content = file_content.decode('utf-8')
        else:
            raise HTTPException(
                status_code=400, detail="Unsupported file type")

        # Store document content globally
        doc_id = str(uuid.uuid4())
        stored_documents.append({
            'id': doc_id,
            'filename': file.filename,
            'content': content,
            'file_type': file.content_type
        })

        # Save to database
        conn = sqlite3.connect('chatbot_memory.db')
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO documents (id, filename, content, file_type)
            VALUES (?, ?, ?, ?)
        ''', (doc_id, file.filename, content, file.content_type))

        conn.commit()
        conn.close()

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}")

    return content


def create_context_from_documents(user_message: str) -> str:
    """Create context from stored documents relevant to user message"""
    if not stored_documents:
        return ""

    # Always include all documents for now, but limit content length
    relevant_docs = []

    for doc in stored_documents:
        # Include first 2000 characters of each document
        content_preview = doc['content'][:2000]
        if len(doc['content']) > 2000:
            content_preview += "..."

        relevant_docs.append(
            f"Document: {doc['filename']}\nContent:\n{content_preview}")

    if relevant_docs:
        return "\n\n" + "="*50 + "\n".join(relevant_docs) + "\n" + "="*50
    return ""

# Chat endpoint


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Receives user message, processes through Gemini via LangChain with document context,
    and returns AI response with conversation memory
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Get or create chat history for this session
        history = get_or_create_history(session_id)

        # Save user message to database
        save_to_db(session_id, "human", request.message)

        # Build the user message with document context if requested
        user_message = request.message

        if request.include_documents and stored_documents:
            # Create document context
            document_context = create_context_from_documents(request.message)
            if document_context:
                user_message = f"{request.message}\n\n{document_context}\n\nPlease answer based on the provided documents if relevant."

        # Initialize LLM and Chain dynamically
        llm = get_llm(request.provider, request.model)
        chain = prompt | llm

        # Create chain with message history
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: get_or_create_history(session_id),
            input_messages_key="input",
            history_messages_key="history"
        )

        # Get response from LangChain
        response = chain_with_history.invoke(
            {"input": user_message},
            config={"configurable": {"session_id": session_id}}
        )

        # Extract text content from response
        response_text = response.content if hasattr(
            response, 'content') else str(response)

        # Save AI response to database
        save_to_db(session_id, "ai", response_text)

        return {
            "reply": response_text,
            "session_id": session_id
        }

    except Exception as e:
        return {"reply": f"Error: {str(e)}"}

# Document upload endpoint


@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document for the chatbot to reference
    """
    try:
        # Validate file type
        allowed_types = ['.pdf', '.txt', '.md']
        if not any(file.filename.endswith(ext) for ext in allowed_types):
            raise HTTPException(
                status_code=400,
                detail="Only PDF, TXT, and MD files are supported"
            )

        # Process document
        content = await process_document(file)

        return DocumentUploadResponse(
            message=f"Document '{file.filename}' uploaded and processed successfully!",
            document_id=str(uuid.uuid4()),
            filename=file.filename
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get conversation history endpoint


@app.get("/history/{session_id}")
async def get_conversation_history(session_id: str):
    """
    Get conversation history for a specific session
    """
    try:
        history = load_conversation_history(session_id)
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Clear conversation endpoint


@app.post("/clear/{session_id}")
async def clear_conversation(session_id: str):
    """
    Clear conversation history for a specific session
    """
    try:
        # Clear from LangChain memory
        if session_id in session_histories:
            del session_histories[session_id]

        # Clear from database
        conn = sqlite3.connect('chatbot_memory.db')
        cursor = conn.cursor()
        cursor.execute(
            'DELETE FROM conversations WHERE session_id = ?', (session_id,))
        conn.commit()
        conn.close()

        return {"message": "Conversation cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get stored documents endpoint


@app.get("/documents")
async def get_documents():
    """
    Get list of stored documents
    """
    return {"documents": [{"filename": doc["filename"], "id": doc["id"]} for doc in stored_documents]}

# Serve the HTML frontend


@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

# Health check


@app.get("/health")
async def health():
    return {"status": "healthy"}


"""
🚀 HOW TO RUN:
1. Install dependencies:
   pip install -r requirements.txt

2. Create .env file with:
   GOOGLE_API_KEY=your_api_key_here

3. Start server:
   uvicorn app:app --reload

4. Open browser:
   http://localhost:8000
"""
