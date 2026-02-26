# app.py - USING OPENAI EMBEDDINGS
from flask import Flask, render_template, request, jsonify, send_from_directory
from src.helper import download_embeddings  # Changed import
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from src.prompt import system_prompt
from dotenv import load_dotenv
import os
import logging
import traceback
import sys
import time

# Setup logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Validate API keys
if not PINECONE_API_KEY:
    logger.error("❌ PINECONE_API_KEY not found")
    raise ValueError("PINECONE_API_KEY is required")

if not OPENAI_API_KEY:
    logger.error("❌ OPENAI_API_KEY not found")
    raise ValueError("OPENAI_API_KEY is required")

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize Flask app
app = Flask(__name__, 
            static_folder='static',
            static_url_path='/static',
            template_folder='templates')

# Global variables
embeddings = None
docsearch = None
retriever = None
chat_model = None
initialized = False
memory_store = {}

def initialize_system():
    """Initialize all components - FAST with OpenAI embeddings"""
    global embeddings, docsearch, retriever, chat_model, initialized
    
    try:
        logger.info("📚 Loading OpenAI embeddings (API-based, no local model)...")
        embeddings = download_embeddings()  # This is now instant
        logger.info("✅ Embeddings ready")
        
        logger.info("🔌 Connecting to Pinecone...")
        index_name = "medi-guide-bot"
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        logger.info("✅ Pinecone connected")
        
        logger.info("⚙️ Setting up retriever...")
        retriever = docsearch.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        logger.info("🤖 Initializing Chat Model...")
        chat_model = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            max_retries=2,
            request_timeout=30
        )
        logger.info("✅ Chat Model ready")
        
        initialized = True
        logger.info("✅ System initialized successfully!")
        
        # Log memory usage
        import psutil
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"📊 Memory usage: {memory_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization error: {str(e)}")
        logger.error(traceback.format_exc())
        initialized = False
        return False

# Initialize on startup
logger.info("🚀 Starting MediGuide AI...")
initialize_system()

@app.route('/')
def index():
    """Render the main chat page"""
    return render_template('chat.html', booting=not initialized)

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

def get_memory(session_id):
    """Get or create conversation memory"""
    if session_id not in memory_store:
        memory_store[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer',
            max_token_limit=500
        )
    return memory_store[session_id]

@app.route("/get", methods=["POST"])
def chat():
    """Handle chat messages"""
    try:
        if not initialized:
            return jsonify({
                "error": "System is starting up (5-10 seconds)",
                "status": "booting"
            }), 503
        
        session_id = request.remote_addr
        msg = request.form.get("msg", "").strip()
        
        if not msg:
            return jsonify({"error": "Empty message"}), 400
        
        memory = get_memory(session_id)
        
        conv_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            retriever=retriever,
            memory=memory,
            return_source_documents=False,
            verbose=False
        )
        
        logger.info(f"💬 Processing: {msg[:30]}...")
        
        result = conv_chain.invoke({"question": msg})
        response = result.get('answer', 'I could not generate a response.')
        
        return str(response)
    
    except Exception as e:
        logger.error(f"❌ Chat error: {str(e)}")
        return jsonify({"error": "Service temporarily unavailable"}), 500

@app.route("/health")
def health():
    """Health check"""
    import psutil
    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
    
    return jsonify({
        "status": "healthy" if initialized else "booting",
        "initialized": initialized,
        "memory_mb": round(memory_mb, 1),
        "active_sessions": len(memory_store),
        "max_sessions": 30,
        "embeddings": embeddings is not None,
        "pinecone": docsearch is not None,
        "chat_model": chat_model is not None
    })

@app.route("/clear", methods=["POST"])
def clear_memory():
    """Clear all sessions"""
    global memory_store
    memory_store = {}
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host="0.0.0.0", port=port, debug=False)