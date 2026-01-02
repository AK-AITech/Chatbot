# 🤖 Multi-Provider RAG Chatbot

A powerful, educational chatbot that demonstrates Retrieval-Augmented Generation (RAG) using LangChain, FastAPI, and multiple LLM providers (Gemini, OpenAI, Anthropic, Ollama, and Hugging Face).

## 🌟 Features

*   **Multi-Provider Support**: Switch between Google Gemini, OpenAI, Anthropic, Ollama (Local), and Hugging Face instantly.
*   **RAG (Retrieval-Augmented Generation)**: Upload PDF, TXT, or MD files and chat with them.
*   **Memory**: Remembers your conversation context.
*   **Modern UI**: Clean, dark-themed interface with markdown support.
*   **FastAPI Backend**: High-performance async Python server.

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
    OLLAMA_BASE_URL=http://localhost:11434  # Default for local Ollama
    ```

3.  **Run the Application**
    ```bash
    python start.py
    ```

4.  **Open in Browser**
    Go to `http://localhost:8000`

## 📚 How to Learn RAG with this Project

This project is designed to be a learning tool for building RAG applications. Here is how you can use it to understand the core concepts:

### 1. The RAG Pipeline
RAG consists of three main steps: **Retrieval**, **Augmentation**, and **Generation**.

*   **Ingestion (Upload)**: Look at `process_document` in `app.py`.
    *   We read the file (PDF/Text).
    *   We extract the text content.
    *   *Learning Task*: Try adding support for `.docx` files using `python-docx`.

*   **Storage**: Look at `stored_documents` and `init_db` in `app.py`.
    *   We store the raw text in memory and SQLite.
    *   *Learning Task*: In a production app, you would chunk this text and store "embeddings" (vector representations) in a Vector Database (like ChromaDB or Pinecone) for semantic search. Try implementing a simple vector search!

*   **Retrieval (Context Creation)**: Look at `create_context_from_documents` in `app.py`.
    *   Currently, we do a "naive" retrieval: we just grab the first 2000 characters of *all* documents.
    *   *Learning Task*: This is where the magic happens. Replace this function with a logic that only selects the *relevant* parts of the documents based on the user's question.

*   **Generation (Chat)**: Look at the `chat` endpoint in `app.py`.
    *   We combine the `user_message` + `document_context` into a single prompt.
    *   We send this to the LLM (Gemini, OpenAI, Anthropic, etc.).
    *   The LLM uses the context to answer the question.

### 2. Multi-Provider Logic
Check the `get_llm` function in `app.py`.
*   It demonstrates how LangChain abstracts different providers (`ChatGoogleGenerativeAI`, `ChatOpenAI`, `ChatAnthropic`, `ChatOllama`, `ChatHuggingFace`) into a common interface.
*   You can easily swap models without changing your core application logic.

### 3. Frontend Integration
Check `index.html`.
*   See how the frontend sends the `provider` and `include_documents` flags to the backend.
*   It handles the chat history and displays the response.

## 🛠️ Tech Stack

*   **Backend**: Python, FastAPI, SQLite
*   **AI/LLM**: LangChain, Google Gemini, OpenAI, Anthropic, Ollama, Hugging Face
*   **Frontend**: HTML, CSS, JavaScript (Vanilla)

## 📝 License

MIT License - Feel free to use this for learning and building your own projects!
