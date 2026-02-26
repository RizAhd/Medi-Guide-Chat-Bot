# api/index.py
from flask import Flask, render_template, request, jsonify, send_from_directory
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from src.prompt import system_prompt
import os
import logging
import traceback
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Initialize Flask app for Vercel
app = Flask(__name__, 
            static_folder='../static',
            template_folder='../templates')

# Global variables
embeddings = None
docsearch = None
retriever = None
chat_model = None
initialized = False

def initialize_system():
    """Initialize all components"""
    global embeddings, docsearch, retriever, chat_model, initialized
    
    try:
        # Get API keys from environment
        PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
        OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
        
        if not PINECONE_API_KEY:
            logger.error("❌ PINECONE_API_KEY not found")
            return False
        
        if not OPENAI_API_KEY:
            logger.error("❌ OPENAI_API_KEY not found")
            return False
        
        os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        
        logger.info("📚 Loading embeddings...")
        embeddings = download_hugging_face_embeddings()
        logger.info("✅ Embeddings loaded")
        
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
        return True
        
    except Exception as e:
        logger.error(f"❌ Init error: {str(e)}")
        return False

# Initialize on startup
initialize_system()

# Memory store (simplified for serverless)
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

@app.route('/')
def index():
    """Render main page"""
    return render_template('chat.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('../static', path)

@app.route('/get', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        if not initialized:
            return "System is starting up. Please wait 10 seconds and try again.", 503
        
        session_id = request.remote_addr or 'anonymous'
        msg = request.form.get("msg", "").strip()
        
        if not msg:
            return "Please enter a message.", 400
        
        memory = get_memory(session_id)
        
        conv_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            retriever=retriever,
            memory=memory,
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": system_prompt}
        )
        
        logger.info(f"💬 Processing: {msg[:30]}...")
        result = conv_chain.invoke({"question": msg})
        response = result.get('answer', 'No response generated')
        
        return str(response)
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return "I'm having trouble right now. Please try again.", 500

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy' if initialized else 'starting',
        'initialized': initialized
    })

@app.route('/clear', methods=['POST'])
def clear():
    """Clear session memory"""
    session_id = request.remote_addr or 'anonymous'
    if session_id in memory_store:
        del memory_store[session_id]
    return jsonify({'status': 'cleared'})

# For Vercel serverless
def handler(request, context):
    return app(request, context)