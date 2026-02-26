# src/helper.py - MEMORY OPTIMIZED
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document
import os
import gc
import torch

# Force CPU and memory optimizations
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
torch.set_num_threads(1)  # Limit CPU threads


def load_pdf_file(data):
    """Load PDF files from directory"""
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Filter documents to keep only essential metadata"""
    minimal_docs = []
    for doc in docs:
        if doc.page_content and len(doc.page_content.strip()) > 10:
            src = doc.metadata.get("source", "unknown")
            minimal_docs.append(
                Document(
                    page_content=doc.page_content.strip(),
                    metadata={"source": src}
                )
            )
    return minimal_docs


def text_split(extracted_data):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


def download_hugging_face_embeddings():
    """Get HuggingFace embeddings model - optimized for memory"""
    try:
        # Clear cache before loading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load model with memory optimizations
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={
                'device': 'cpu',
                'low_cpu_mem_usage': True
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32  # Smaller batch size
            }
        )
        
        # Force garbage collection after loading
        gc.collect()
        
        return embeddings
        
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        # Fallback to a lighter model if needed
        embeddings = HuggingFaceEmbeddings(
            model_name='all-MiniLM-L6-v2',  # Even lighter
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings