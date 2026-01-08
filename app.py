from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)
load_dotenv()

# Load API keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize embeddings
embeddings = download_hugging_face_embeddings()

# Connect to Pinecone index
index_name = "medi-guide-bot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Retriever for RAG
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Chat model
chat_model = ChatOpenAI(model="gpt-4o", temperature=0.3)

# Memory store per session
memory_store = {}

def get_memory(session_id):
    if session_id not in memory_store:
        # Keeps full conversation for session
        memory_store[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return memory_store[session_id]

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    session_id = request.remote_addr
    msg = request.form["msg"]

    memory = get_memory(session_id)

    # ConversationalRetrievalChain handles RAG + memory
    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=retriever,
        memory=memory,
        return_source_documents=False
    )

    # Run the chain
    result = conv_chain.run(msg)

    return result

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
