from flask import Flask, render_template, request, jsonify
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from src.prompt import *
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Load API keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Validate API keys
if not PINECONE_API_KEY or not OPENAI_API_KEY:
    logger.error("Missing API keys. Please check your environment variables.")
    raise ValueError("PINECONE_API_KEY and OPENAI_API_KEY must be set")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize embeddings
try:
    embeddings = download_hugging_face_embeddings()
    logger.info("Embeddings loaded successfully")
except Exception as e:
    logger.error(f"Error loading embeddings: {e}")
    raise

# Initialize Pinecone vector store
index_name = "medi-guide-bot"
try:
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    logger.info("Pinecone vector store initialized successfully")
except Exception as e:
    logger.error(f"Error connecting to Pinecone: {e}")
    raise

# Create retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Initialize chat model
chat_model = ChatOpenAI(
    model="gpt-4o", 
    temperature=0.3,
    max_retries=2
)

# Memory store for conversations
memory_store = {}

def get_memory(session_id):
    """Get or create conversation memory for a session"""
    if session_id not in memory_store:
        memory_store[session_id] = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key='answer'
        )
    return memory_store[session_id]

@app.route("/")
def index():
    """Render the chat interface"""
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    """Handle chat messages"""
    try:
        session_id = request.remote_addr or request.headers.get('X-Forwarded-For', 'unknown')
        msg = request.form["msg"]
        
        if not msg or not msg.strip():
            return jsonify({"error": "Empty message"}), 400
        
        logger.info(f"Received message from {session_id}: {msg[:50]}...")
        
        memory = get_memory(session_id)
        
        # Create conversation chain
        conv_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            retriever=retriever,
            memory=memory,
            return_source_documents=False,
            verbose=False
        )
        
        # Get response
        result = conv_chain.run(msg)
        logger.info(f"Response sent to {session_id}")
        
        return str(result)
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return "I apologize, but I'm experiencing technical difficulties. Please try again in a moment."

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Render"""
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)