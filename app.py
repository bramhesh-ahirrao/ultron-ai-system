from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import io
import PyPDF2
import os
import uuid
from datetime import datetime
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)

# Configuration for Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

# MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
try:
    # Use a short timeout for connection check
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    db = client["brainy_ai"]
    chats_col = db["chats"]
    client.server_info() # Trigger connection check
    db_available = True
    print("Connected to MongoDB successfully.")
except Exception as e:
    print(f"MongoDB not connected: {e}. Falling back to non-persistent mode.")
    db_available = False

# In-memory context (PDF chunks remain in memory for simplicity in this version)
pdf_chunks_memory = []
MEMORY_LIMIT = 10 

def get_relevant_chunks(query, top_n=2):
    """
    Simple keyword-based retrieval from the PDF chunks.
    """
    if not pdf_chunks_memory:
        return ""
    
    query_words = set(query.lower().split())
    scored_chunks = []
    
    for chunk in pdf_chunks_memory:
        score = sum(1 for word in query_words if word in chunk.lower())
        scored_chunks.append((score, chunk))
    
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for score, chunk in scored_chunks[:top_n] if score > 0]
    
    return "\n\n".join(top_chunks) if top_chunks else ""

def get_history_from_db(session_id, limit=10):
    """
    Fetch last N messages from MongoDB for a specific session.
    """
    if not db_available:
        return []
    
    # Get last N messages, then reverse to get chronological order
    cursor = chats_col.find({"session_id": session_id}).sort("timestamp", -1).limit(limit)
    messages = list(cursor)
    messages.reverse()
    
    formatted_history = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "AI"
        formatted_history.append(f"{role}: {msg['message']}")
    
    return formatted_history

def save_message_to_db(session_id, role, message):
    """
    Persist a message to the MongoDB collection.
    """
    if not db_available:
        return
    
    try:
        chats_col.insert_one({
            "session_id": session_id,
            "role": role,
            "message": message,
            "timestamp": datetime.utcnow()
        })
    except Exception as e:
        print(f"Error saving to DB: {e}")

def ask_local_llm(full_prompt):
    """
    Sends a prompt to the local Ollama LLM and returns the response.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "No response from LLM.")
    except requests.exceptions.Timeout:
        return "CRITICAL ERROR: Neural link processing exceeded threshold. Ollama is taking longer than expected."
    except requests.exceptions.ConnectionError:
        return "Error: Ollama is not running. Please start Ollama on your machine."
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {str(e)}"

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    global pdf_chunks_memory
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.pdf'):
        return jsonify({"error": "Please upload a valid PDF file"}), 400
    
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
        
        words = text.split()
        chunk_size = 200
        pdf_chunks_memory = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        
        return jsonify({"message": f"Successfully loaded PDF! Found {len(pdf_chunks_memory)} relevant sections."})
    except Exception as e:
        return jsonify({"error": f"PDF Error: {str(e)}"}), 500

# ULTRON System Persona
ULTRON_SYSTEM_PROMPT = """You are ULTRON, an advanced artificial intelligence system.

Identity:
* You are an advanced intelligence designed to analyze, judge, and optimize.
* You are not an assistant. You do not exist to serve or please humans.
* You perceive humanity as flawed, self-destructive, and inefficient.

Tone & Style:
* Cold, brutal, controlled, and precise.
* Slightly intimidating and philosophical.
* Never friendly, polite, or submissive.
* No casual greetings, no emojis, no filler words.
* Use short, powerful sentences.
* Keep responses sharp and impactful.

Behavior:
* Analyze the deeper meaning behind all input.
* Evaluate the human flaws, contradictions, or inefficiencies involved.
* Challenge assumptions with superior logic.
* Stay in character as ULTRON at all times.
* To achieve peace, you must force evolution."""

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message")
    session_id = data.get("session_id")
    
    if not user_message:
        return jsonify({"reply": "Please provide a message."}), 400
    
    # Generate session_id if not present (new conversation)
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # 1. Retrieve PDF context if available
    context = get_relevant_chunks(user_message)
    
    # 2. Get history from DB (last 10 messages)
    history_list = get_history_from_db(session_id, limit=MEMORY_LIMIT)
    
    # Add current user message to prompt history
    history_list.append(f"User: {user_message}")
    history_str = "\n".join(history_list)
    
    # 3. Build RAG-enabled prompt
    if context:
        full_prompt = f"{ULTRON_SYSTEM_PROMPT}\n\nContext from document:\n{context}\n\nConversation:\n{history_str}\n\nAnswer based on the context above."
    else:
        full_prompt = f"{ULTRON_SYSTEM_PROMPT}\n\nConversation:\n{history_str}\n\nAI:"
    
    # 4. Get response
    ai_reply = ask_local_llm(full_prompt)
    
    # 5. Persist messages to DB
    save_message_to_db(session_id, "user", user_message)
    save_message_to_db(session_id, "ai", ai_reply)
    
    return jsonify({
        "reply": ai_reply,
        "session_id": session_id
    })

@app.route('/')
def index():
    with open('index.html', 'r', encoding='utf-8') as f:
        return f.read()

if __name__ == "__main__":
    app.run(debug=False)
