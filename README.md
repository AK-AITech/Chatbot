# 🤖 Multi-Provider Advanced RAG Chatbot (LangGraph)

A sophisticated, educational chatbot that demonstrates **Advanced Retrieval-Augmented Generation (RAG)** and **Agentic Workflows** using LangGraph, ChromaDB, FastAPI, and multiple LLM providers.

## 🌟 Features

*   **Advanced RAG**: Uses **ChromaDB** vector store and **HuggingFace Embeddings** for semantic search.
*   **Agentic Workflow (LangGraph)**: Orchestrates retrieval, web search, and generation using a state machine.
*   **Multi-Provider Support**: Switch between Google Gemini, OpenAI, Anthropic, Ollama (Local), and Hugging Face instantly.
*   **Web Search**: Integrated real-time web search via DuckDuckGo.
*   **Memory**: Persistent conversation history stored in SQLite.
*   **Modern UI**: Clean, dark-themed interface with markdown support.

## 🚀 Quick Start

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Environment**
    Create a `.env` file in the root directory:
    ```env
    GOOGLE_API_KEY=your_gemini_api_key
    OPENAI_API_KEY=your_openai_api_key
    ANTHROPIC_API_KEY=your_anthropic_api_key
    HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
    OLLAMA_BASE_URL=http://localhost:11434
    ```

3.  **Run the Application**
    ```bash
    python start.py
    ```

4.  **Open in Browser**
    Go to `http://localhost:8000`

## 📚 How to Learn Advanced RAG & LangGraph

This project is a blueprint for modern AI applications. Here is how to study it:

### 1. Advanced RAG Pipeline
Unlike basic RAG, Advanced RAG uses semantic search to find relevant information.

*   **Ingestion & Chunking**: Look at `process_document` in `app.py`.
    *   We use `RecursiveCharacterTextSplitter` to break documents into 1000-character chunks.
    *   We use `HuggingFaceEmbeddings` to turn text into vectors.
    *   We store these in **ChromaDB**.
*   **Semantic Retrieval**: Look at `retrieve_node` in `app.py`.
    *   Instead of reading the whole file, we use `vector_store.similarity_search` to find the top 3 most relevant chunks for the user's specific question.

### 2. Agentic Workflows with LangGraph
The application logic is no longer a simple linear script; it's a **Graph**.

*   **State Management**: Look at `AgentState` in `app.py`. It tracks the input, context, and history throughout the process.
*   **Nodes**: Each step (Retrieve, Web Search, Generate) is a separate node in the graph.
*   **Edges**: Look at the `workflow` construction. It defines the order of operations:
    1.  `retrieve` -> `web_search` -> `generate`
*   *Learning Task*: Try adding a "conditional edge" that only performs a web search if the retrieved documents don't contain the answer!

### 3. Multi-Provider Abstraction
Check the `get_llm` function. It shows how LangChain allows you to swap the "brain" of your agent (Gemini, GPT-4o, Claude) without changing the graph logic.

## 🛠️ Tech Stack

*   **Orchestration**: LangGraph, LangChain
*   **Vector Store**: ChromaDB
*   **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
*   **Backend**: FastAPI, SQLite
*   **Frontend**: Vanilla JS, CSS, HTML

## 📝 License

MIT License - Feel free to use this for learning and building your own projects!
