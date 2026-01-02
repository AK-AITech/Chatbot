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
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
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
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not found in environment variables")
        
        return ChatAnthropic(
            model=model or "claude-3-5-sonnet-20240620",
            anthropic_api_key=api_key,
            temperature=0.2
        )
    elif provider == "ollama":
        return ChatOllama(
            model=model or "llama3",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.2
        )
    elif provider == "huggingface":
        api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not api_key:
            raise HTTPException(status_code=400, detail="HUGGINGFACEHUB_API_TOKEN not found in environment variables")
        
        llm = HuggingFaceEndpoint(
            repo_id=model or "mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=api_key,
            temperature=0.2
        )
        return ChatHuggingFace(llm=llm)
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

# Initialize Advanced RAG components
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    collection_name="chatbot_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Global variables for conversation management
session_histories = {}  # Store chat histories per session
stored_documents = []

# Request models


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    include_documents: bool = True
    use_web_search: bool = False
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

        # Advanced RAG: Chunk and add to Vector Store
        chunks = text_splitter.split_text(content)
        vector_store.add_texts(
            texts=chunks,
            metadatas=[{"filename": file.filename, "doc_id": doc_id} for _ in chunks]
        )

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


# LangGraph State Definition
class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    context: str
    use_web_search: bool
    include_documents: bool
    provider: str
    model: Optional[str]
    response: str

# Node Functions
def retrieve_node(state: AgentState):
    """Retrieve relevant documents from ChromaDB"""
    if not state["include_documents"]:
        return {"context": state.get("context", "")}
    
    query = state["input"]
    # Search for top 3 relevant chunks
    docs = vector_store.similarity_search(query, k=3)
    
    context = state.get("context", "")
    if docs:
        doc_context = "\n\nRelevant Document Chunks:\n" + "\n".join([f"[{d.metadata['filename']}]: {d.page_content}" for d in docs])
        context += doc_context
        
    return {"context": context}

def web_search_node(state: AgentState):
    """Perform web search if enabled"""
    if not state["use_web_search"]:
        return {"context": state.get("context", "")}
    
    query = state["input"]
    try:
        search = DuckDuckGoSearchRun()
        results = search.run(query)
        web_context = f"\n\nWeb Search Results:\n{results}"
        return {"context": state.get("context", "") + web_context}
    except Exception as e:
        return {"context": state.get("context", "") + f"\n\nWeb Search Error: {str(e)}"}

def generate_node(state: AgentState):
    """Generate response using the selected LLM"""
    llm = get_llm(state["provider"], state["model"])
    
    # Prepare the prompt with context
    system_msg = """You are a precise and helpful AI assistant.
Use the provided context (documents and web search) to answer the user's question.
If the context doesn't contain the answer, use your general knowledge but mention that it's not in the provided documents.
Stay professional and concise."""
    
    full_input = f"Context:\n{state['context']}\n\nQuestion: {state['input']}"
    
    messages = [("system", system_msg)]
    # Add history
    for msg in state["chat_history"]:
        if isinstance(msg, HumanMessage):
            messages.append(("human", msg.content))
        elif isinstance(msg, AIMessage):
            messages.append(("ai", msg.content))
            
    messages.append(("human", full_input))
    
    response = llm.invoke(messages)
    return {"response": response.content}

# Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app_graph = workflow.compile()

# Chat endpoint


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Receives user message, processes through LangGraph with Advanced RAG,
    and returns AI response with conversation memory
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Get or create chat history for this session
        history_obj = get_or_create_history(session_id)
        chat_history = history_obj.messages

        # Save user message to database
        save_to_db(session_id, "human", request.message)

        # Run the LangGraph workflow
        initial_state = {
            "input": request.message,
            "chat_history": chat_history,
            "context": "",
            "use_web_search": request.use_web_search,
            "include_documents": request.include_documents,
            "provider": request.provider,
            "model": request.model,
            "response": ""
        }

        result = app_graph.invoke(initial_state)
        response_text = result["response"]

        # Update LangChain memory
        history_obj.add_user_message(request.message)
        history_obj.add_ai_message(response_text)

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
