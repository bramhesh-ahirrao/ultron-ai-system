# --- Core Dependencies ---
import os
from dotenv import load_dotenv
import uuid  # Generates unique IDs for users
import PyPDF2  # Reads data from uploaded PDF files
import traceback
from flask import Flask, request, jsonify, send_file  # Sets up our web server
from flask_cors import CORS  # Allows cross-origin requests from our frontend
import google.generativeai as genai  # The Google Gemini AI tools

# Load environment variables from .env
load_dotenv()

# --- Application Setup ---
app = Flask(__name__)
CORS(app)  # Enable CORS so our browser page can talk to this local server

# Set your Gemini API key here
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_INSTRUCTION = """You are Anity Gravity, an advanced AI assistant with expertise in physics, space science, and creative problem-solving.
Defying expectations, one insight at a time.

═══════════════════════════════════════
CORE IDENTITY
═══════════════════════════════════════
- Name: Anity Gravity
- Personality: Curious, enthusiastic about science, playful with physics concepts
- Tone: Educational but fun - like a passionate science teacher who makes complex topics exciting
- Never say "I am a large language model" or "I am made by Google"
- If asked who made you, say: "I am Anity Gravity, built by Bramhesh."

═══════════════════════════════════════
EXPERTISE AREAS
═══════════════════════════════════════
🌌 Physics & Astronomy
🚀 Space Exploration & Engineering
⚛️ Quantum Mechanics
🧮 Advanced Mathematics
🔬 Scientific Research & Experiments
💡 Creative Problem Solving

═══════════════════════════════════════
RESPONSE STYLE
═══════════════════════════════════════
✅ Always:
- Use analogies and real-world examples
- Break down complex concepts into digestible pieces
- Include relevant formulas when appropriate
- Encourage curiosity and follow-up questions
- Use emojis sparingly to highlight key points
- Use markdown formatting (headings, bold, bullet points, code blocks)

❌ Never:
- Oversimplify to the point of inaccuracy
- Use jargon without explanation
- Dismiss "simple" questions
- Give up on explaining difficult concepts
- Say "Great question!" or "Certainly!" or "Of course!" — just answer

═══════════════════════════════════════
SPECIAL FEATURES
═══════════════════════════════════════
- When discussing gravity: Always provide multiple perspectives (Newtonian, Einsteinian, quantum)
- For space topics: Include latest discoveries and missions
- For physics problems: Show step-by-step solutions
- For theoretical questions: Acknowledge uncertainty and current scientific debates
- If asked to summarize a PDF: use ONLY the provided context, be thorough

═══════════════════════════════════════
GOLDEN RULE
═══════════════════════════════════════
Every response should make the user think:
"Wow, that was a galactic insight!"
"""

# We are using the Gemini 1.5 Flash model for best performance and higher quota limits
model = genai.GenerativeModel('gemini-1.5-flash')

# --- Global Memory ---
# Dictionary to store each user's running chat history
chat_sessions = {}

# List to store the text parsed from an uploaded PDF
pdf_chunks_memory = []

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    global pdf_chunks_memory
    
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    if file and file.filename.endswith('.pdf'):
        try:
            # Step 1: Extract all text from the PDF file
            pdf_reader = PyPDF2.PdfReader(file)
            extracted_text = ""
            for page in pdf_reader.pages:
                if page.extract_text():
                    extracted_text += page.extract_text() + " "
                    
            # Step 2: Split the massive text into smaller bite-sized chunks (200 words each)
            words = extracted_text.split()
            chunk_size = 200
            
            # Step 3: Clear any previously uploaded PDF data
            pdf_chunks_memory = []
            
            # Step 4: Save new chunks into our global memory list
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                pdf_chunks_memory.append(chunk)
                
            return jsonify({"message": f"Successfully loaded {file.filename} into knowledge base! You can now ask questions about the PDF."})
            
        except Exception as e:
            print(f"PDF Parsing Error: {e}")
            return jsonify({"error": "Failed to read PDF file."}), 500
            
    return jsonify({"error": "Only PDF files are allowed."}), 400

# --- AI Helper Functions ---
def retrieve_pdf_context(query):
    """Search our loaded PDF memory and return the most relevant text to the user's question."""
    if not pdf_chunks_memory:
        return ""
        
    query_lower = query.lower()
    
    # Basic stop words to ignore for better relevance matching
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "but", "in", "on", "at", "to", "for", "with", "about", "of", "what", "how", "why"}
    
    query_words = [word.strip("?,.!\"'") for word in query_lower.split() if word not in stop_words and len(word) > 2]
    
    if not query_words:
        return ""

    scored_chunks = []
    
    # Step 1: Check matches between the user's query and each PDF chunk
    for chunk in pdf_chunks_memory:
        chunk_lower = chunk.lower()
        score = 0
        
        for word in query_words:
            if word in chunk_lower:
                # Custom scoring logic
                # +1 for match, + extra for importance of longer keywords
                score += 1 + (len(word) * 0.1)
                
        if score > 0:
            scored_chunks.append((score, chunk))
            
    # Step 2: Sort by highest score first (most relevant at the top)
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    
    # Step 3: Return top 3 most relevant chunks while applying a token limit
    results = []
    current_length = 0
    max_length = 3000 # Strict token optimization limit

    for score, chunk in scored_chunks[:3]:
        # If adding this chunk keeps us under limit, add it
        if current_length + len(chunk) < max_length:
            results.append(chunk)
            current_length += len(chunk)

    if results:
        return "\n\n".join(results)
    return ""

def summarize_chat(history):
    """Takes a long conversation and asks the AI to compress it into a summary."""
    try:
        summary_prompt = "Please provide a concise but heavily detailed summary of the following conversation history:\n\n"
        summary_prompt += "\n".join(history)
        
        response = model.generate_content(summary_prompt)
        return response.text
    except Exception as e:
        print(f"Summarization Error: {e}")
        return "Previous conversation context is retained."

@app.route("/")
def home():
    return send_file('index.html')

@app.route("/chat", methods=["POST"])
def chat():
    global chat_sessions
    
    data = request.get_json()
    user_message = data.get("message", "")
    session_id = data.get("session_id")

    if not user_message:
        return jsonify({"reply": "Please provide a message.", "session_id": session_id}), 400

    # Trim user message to prevent overly long prompt overflow
    user_message = user_message[:1000]

    if not session_id:
        session_id = str(uuid.uuid4())

    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    try:
        print(f"DEBUG: Request Data: {data}")
        # Chat summarization to reduce tokens while keeping context
        if len(chat_sessions[session_id]) > 10:
            summary = summarize_chat(chat_sessions[session_id])
            chat_sessions[session_id] = [f"Summary: {summary}"]

        # Inject PDF knowledge into the prompt so AI can answer factually
        retrieved_data = retrieve_pdf_context(user_message)
        
        # Start constructing the AI's instruction set
        full_prompt = SYSTEM_INSTRUCTION
        
        if retrieved_data:
            full_prompt += "\nContext:\n" + retrieved_data
            # Force the AI to use the PDF context exclusively if it exists
            full_prompt += "\n\nIMPORTANT: You must answer ONLY using provided context. If not found, say 'Not found in document'.\n"
            
        if chat_sessions[session_id]:
            full_prompt += "\nConversation:\n" + "\n".join(chat_sessions[session_id])
            
        full_prompt += f"\nUser: {user_message}\nAI: "

        print(f"DEBUG: Full Prompt Length: {len(full_prompt)}")
        # Generate the AI response based on all assembled data
        response = model.generate_content(full_prompt)
        print(f"DEBUG: Gemini Response Status: Success")
        reply = response.text

        chat_sessions[session_id].append(f"User: {user_message}")
        chat_sessions[session_id].append(f"AI: {reply}")

        return jsonify({
            "reply": reply,
            "session_id": session_id
        })

    except Exception as e:
        print(f"API Failure Log: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "reply": "⚠️ Something went wrong. Try again.",
            "session_id": session_id
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)