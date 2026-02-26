# app.py - FULLY OPTIMIZED
from flask import Flask, render_template, request, jsonify, send_from_directory
from src.helper import download_hugging_face_embeddings
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
import gc
import psutil
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
MAX_SESSIONS = 30  # Limit concurrent sessions
MEMORY_LIMIT_MB = 400  # Target memory usage

def log_memory(stage=""):
    """Log current memory usage"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"📊 Memory {stage}: {memory_mb:.1f} MB")
    return memory_mb

def check_memory_limit():
    """Check if we're near memory limit and clean up if needed"""
    memory_mb = log_memory()
    if memory_mb > MEMORY_LIMIT_MB:
        logger.warning(f"⚠️ Memory high ({memory_mb:.1f} MB), cleaning up...")
        gc.collect()
        # Clear old sessions if needed
        if len(memory_store) > 10:
            memory_store.clear()
            logger.info("🧹 Cleared all sessions to free memory")
        return True
    return False

def initialize_system():
    """Initialize all components with memory management"""
    global embeddings, docsearch, retriever, chat_model, initialized
    
    try:
        log_memory("before initialization")
        
        logger.info("📚 Loading embeddings (this may take a moment)...")
        embeddings = download_hugging_face_embeddings()
        log_memory("after embeddings")
        
        # Check if we're still within limits
        if check_memory_limit():
            logger.warning("⚠️ Near memory limit after embeddings")
        
        logger.info("🔌 Connecting to Pinecone...")
        index_name = "medi-guide-bot"
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        log_memory("after pinecone")
        
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
        log_memory("after chat model")
        
        initialized = True
        logger.info("✅ System initialized successfully!")
        log_memory("final")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization error: {str(e)}")
        logger.error(traceback.format_exc())
        initialized = False
        return False

# Initialize on startup with timeout
logger.info("🚀 Starting MediGuide AI...")
try:
    # Set a timeout for initialization
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Initialization timed out")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)  # 60 second timeout
    
    initialize_system()
    signal.alarm(0)  # Disable alarm
    
except TimeoutError:
    logger.error("❌ Initialization timed out (60s)")
    initialized = False
except Exception as e:
    logger.error(f"❌ Startup error: {e}")
    initialized = False

@app.route('/')
def index():
    """Render the main chat page"""
    if not initialized:
        return render_template('chat.html', booting=True)
    return render_template('chat.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

def get_memory(session_id):
    """Get or create conversation memory with size limits"""
    # Clean up if we have too many sessions
    if len(memory_store) >= MAX_SESSIONS:
        # Remove oldest session
        oldest = next(iter(memory_store))
        del memory_store[oldest]
        gc.collect()
        logger.info(f"🧹 Removed oldest session, now {len(memory_store)} active")
    
    if session_id not in memory_store:
        memory_store[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer',
            max_token_limit=500  # Limit memory size
        )
    
    return memory_store[session_id]

@app.route("/get", methods=["POST"])
def chat():
    """Handle chat messages"""
    try:
        # Check memory before processing
        check_memory_limit()
        
        if not initialized:
            return jsonify({
                "error": "System is starting up. Please wait 30 seconds and try again.",
                "status": "booting"
            }), 503
        
        session_id = request.remote_addr
        msg = request.form.get("msg", "").strip()
        
        if not msg:
            return jsonify({"error": "Empty message"}), 400
        
        # Get or create memory
        memory = get_memory(session_id)
        
        # Create conversation chain
        conv_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            retriever=retriever,
            memory=memory,
            return_source_documents=False,
            verbose=False
        )
        
        logger.info(f"💬 Processing: {msg[:30]}...")
        
        # Get response with timeout
        try:
            result = conv_chain.invoke({"question": msg})
            response = result.get('answer', 'I could not generate a response.')
        except Exception as e:
            logger.error(f"Chain error: {str(e)}")
            response = "I'm having trouble. Please try again."
        
        # Clean up after response
        gc.collect()
        
        return str(response)
    
    except Exception as e:
        logger.error(f"❌ Chat error: {str(e)}")
        return jsonify({"error": "Service temporarily unavailable"}), 500

@app.route("/health")
def health():
    """Health check with memory info"""
    memory_mb = log_memory("health check")
    
    return jsonify({
        "status": "healthy" if initialized else "booting",
        "initialized": initialized,
        "memory_mb": round(memory_mb, 1),
        "active_sessions": len(memory_store),
        "max_sessions": MAX_SESSIONS,
        "embeddings": embeddings is not None,
        "pinecone": docsearch is not None,
        "chat_model": chat_model is not None
    })

@app.route("/clear", methods=["POST"])
def clear_memory():
    """Clear all sessions to free memory"""
    global memory_store
    memory_store = {}
    gc.collect()
    logger.info("🧹 Cleared all sessions")
    return jsonify({
        "status": "cleared",
        "memory_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host="0.0.0.0", port=port, debug=False)