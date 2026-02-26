from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, text_split, filter_to_minimal_docs, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def create_and_upload_index():
    """Run this locally to create your Pinecone index - NOT on Render"""
    
    logger.info("Loading PDF files...")
    extracted_data = load_pdf_file(data='data')
    filter_data = filter_to_minimal_docs(extracted_data)
    text_chunks = text_split(filter_data)
    logger.info(f"Created {len(text_chunks)} text chunks")
    
    logger.info("Loading embeddings model...")
    embeddings = download_hugging_face_embeddings()
    
    logger.info("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "medi-guide-bot"
    
    # Delete existing index if you want to recreate (optional)
    # if index_name in pc.list_indexes().names():
    #     pc.delete_index(index_name)
    #     logger.info(f"Deleted existing index {index_name}")
    
    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        logger.info(f"Creating index {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        logger.info("Index created successfully")
    else:
        logger.info(f"Index {index_name} already exists")
    
    # Upload in batches
    batch_size = 80
    total_batches = (len(text_chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        current_batch = i // batch_size + 1
        logger.info(f"Uploading batch {current_batch}/{total_batches}...")
        
        PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            index_name=index_name
        )
    
    logger.info("All documents uploaded successfully!")

if __name__ == "__main__":
    create_and_upload_index()