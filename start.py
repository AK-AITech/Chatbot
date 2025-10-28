"""
Quick Start Script for Gemini Chatbot
Launches the FastAPI server automatically
"""

import os
import sys
import subprocess
from pathlib import Path


def check_env_file():
    """Check if .env file exists and has API key"""
    env_path = Path(".env")

    if not env_path.exists():
        print("❌ Error: .env file not found!")
        print("\n📝 Please create a .env file with your Google API key:")
        print("   GOOGLE_API_KEY=your_api_key_here")
        print("\n🔑 Get your API key from: https://makersuite.google.com/app/apikey")
        return False

    # Read and check if API key is set
    with open(env_path, 'r') as f:
        content = f.read()
        if 'your_gemini_api_key_here' in content or 'your_api_key_here' in content:
            print("⚠️  Warning: Please update your .env file with a real API key!")
            print("🔑 Get your API key from: https://makersuite.google.com/app/apikey")
            return False

        if not content.strip() or 'GOOGLE_API_KEY=' not in content:
            print("❌ Error: GOOGLE_API_KEY not found in .env file!")
            return False

    return True


def main():
    """Main function to start the server"""
    print("🤖 Starting Gemini Chatbot Server...")
    print("=" * 50)

    # Check environment file
    if not check_env_file():
        sys.exit(1)

    print("✅ Environment configuration OK")
    print("🚀 Launching server on http://localhost:8000")
    print("=" * 50)
    print("\n📌 Press Ctrl+C to stop the server\n")

    # Start uvicorn server
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "app:app",
            "--reload",
            "--host", "localhost",
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("\n💡 Make sure you've installed dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
