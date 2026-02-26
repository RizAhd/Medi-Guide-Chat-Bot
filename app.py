# app.py - FIXED VERSION
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

# Initialize Flask app with correct static folder
app = Flask(__name__, 
            static_folder='static',
            static_url_path='/static',
            template_folder='templates')

# Global variables
embeddings = None
docsearch = None
retriever = None
chat_model = None

def initialize_system():
    """Initialize all components with proper error handling"""
    global embeddings, docsearch, retriever, chat_model
    
    try:
        logger.info("📚 Loading embeddings...")
        embeddings = download_hugging_face_embeddings()
        logger.info("✅ Embeddings loaded successfully")
        
        logger.info("🔌 Connecting to Pinecone...")
        index_name = "medi-guide-bot"
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        logger.info("✅ Pinecone connection successful")
        
        logger.info("⚙️ Setting up retriever...")
        retriever = docsearch.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        logger.info("🤖 Initializing Chat Model...")
        # Fixed ChatOpenAI initialization
        chat_model = ChatOpenAI(
            model="gpt-3.5-turbo",  # Changed from gpt-4o to gpt-3.5-turbo for stability
            temperature=0.3,
            max_retries=2,
            request_timeout=60,
            openai_api_key=OPENAI_API_KEY
        )
        logger.info("✅ Chat Model initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization error: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Initialize on startup
initialized = initialize_system()

@app.route('/')
def index():
    """Render the main chat page"""
    return render_template('chat.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Explicitly serve static files"""
    return send_from_directory('static', path)

@app.route("/get", methods=["POST"])
def chat():
    """Handle chat messages"""
    try:
        # Check if system is initialized
        if not initialized or not retriever or not chat_model:
            logger.warning("⚠️ System not fully initialized")
            return jsonify({"error": "System is initializing. Please try again in 30 seconds."}), 503
        
        session_id = request.remote_addr
        msg = request.form.get("msg", "").strip()
        
        if not msg:
            return jsonify({"error": "Empty message"}), 400
        
        # Get or create memory
        memory = get_memory(session_id)
        
        # Create conversation chain with proper configuration
        conv_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            retriever=retriever,
            memory=memory,
            return_source_documents=False,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": system_prompt}
        )
        
        logger.info(f"💬 Processing query: {msg[:50]}...")
        
        # Get response with error handling
        try:
            result = conv_chain.invoke({"question": msg})
            response = result.get('answer', 'I could not generate a response.')
        except Exception as e:
            logger.error(f"Chain invocation error: {str(e)}")
            response = "I'm having trouble processing your request. Please try again."
        
        logger.info(f"✅ Response sent: {response[:50]}...")
        return str(response)
    
    except Exception as e:
        logger.error(f"❌ Chat error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "An error occurred"}), 500

# Memory store
memory_store = {}

def get_memory(session_id):
    """Get or create conversation memory"""
    if session_id not in memory_store:
        memory_store[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
    return memory_store[session_id]

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if initialized else "initializing",
        "initialized": initialized,
        "embeddings": embeddings is not None,
        "pinecone": docsearch is not None,
        "chat_model": chat_model is not None,
        "retriever": retriever is not None
    })

@app.route("/clear", methods=["POST"])
def clear_memory():
    """Clear conversation memory"""
    session_id = request.remote_addr
    if session_id in memory_store:
        del memory_store[session_id]
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host="0.0.0.0", port=port, debug=False)