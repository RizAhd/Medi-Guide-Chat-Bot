from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, text_split
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs = []
    for doc in docs:
        if doc.page_content and len(doc.page_content.strip()) > 10:
            minimal_docs.append(
                Document(
                    page_content=doc.page_content.strip(),
                    metadata=doc.metadata
                )
            )
    return minimal_docs

extracted_data = load_pdf_files(data='data')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medi--guide-bot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# VERY small batch size to avoid 2MB limit
manual_batch_size = 80  # safest
docsearch = None

for i in range(0, len(text_chunks), manual_batch_size):
    batch = text_chunks[i:i + manual_batch_size]

    # Upload this batch to Pinecone
    batch_store = PineconeVectorStore.from_documents(
        documents=batch,
        embedding=embeddings,
        index_name=index_name
    )

    # Merge into main store
    if docsearch is None:
        docsearch = batch_store
    
print("All documents uploaded. Vector store ready.")
