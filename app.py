# app.py
from flask import Flask, render_template, request, jsonify
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from src.prompt import *
from dotenv import load_dotenv
import os
import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
load_dotenv()

# Load API keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

app.secret_key = SECRET_KEY

# Check if API keys are set
if not PINECONE_API_KEY or not OPENAI_API_KEY:
    logger.error("Missing API keys. Please check your environment variables.")
    # Don't crash, but log error - Render will show in logs

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY or ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or ""

# Initialize components with error handling
embeddings = None
docsearch = None
retriever = None
chat_model = None

try:
    logger.info("Loading embeddings...")
    embeddings = download_hugging_face_embeddings()
    logger.info("Embeddings loaded successfully")
    
    logger.info("Connecting to Pinecone...")
    index_name = "medi-guide-bot"
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    logger.info("Pinecone connection successful")
    
    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    logger.info("Initializing Chat Model...")
    chat_model = ChatOpenAI(
        model="gpt-4o", 
        temperature=0.3
    )
    logger.info("Chat Model initialized successfully")
    
except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    logger.error(traceback.format_exc())

# Memory store for conversations
memory_store = {}

def get_memory(session_id):
    if session_id not in memory_store:
        memory_store[session_id] = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )
    return memory_store[session_id]

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    try:
        session_id = request.remote_addr
        msg = request.form.get("msg", "").strip()
        
        if not msg:
            return jsonify({"error": "Empty message"}), 400
        
        if not retriever or not chat_model:
            return jsonify({"error": "System not fully initialized. Please check API keys."}), 503
        
        memory = get_memory(session_id)
        
        conv_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            retriever=retriever,
            memory=memory,
            return_source_documents=False
        )
        
        # Run the chain
        result = conv_chain.run(msg)
        
        return str(result)
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "An error occurred processing your request"}), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Render"""
    return jsonify({
        "status": "healthy",
        "embeddings": embeddings is not None,
        "pinecone": docsearch is not None,
        "chat_model": chat_model is not None,
        "api_keys_set": bool(PINECONE_API_KEY and OPENAI_API_KEY)
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host="0.0.0.0", port=port, debug=False)