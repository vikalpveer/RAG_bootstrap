"""
Document Ingestion Script for RAG System

This script demonstrates the key steps in building a RAG system:
1. Loading documents from markdown files
2. Splitting documents into chunks
3. Creating embeddings for each chunk
4. Storing embeddings in a vector database (ChromaDB)

Understanding Embeddings:
- Embeddings are numerical representations (vectors) of text
- Similar text has similar vector representations
- This allows semantic search (finding meaning, not just keywords)
"""

import os
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm

from utils import print_header, print_separator, display_rag_stats


# Configuration
DATA_DIR = "../data"
VECTORSTORE_DIR = "../vectorstore"
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks to maintain context
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, efficient model


def load_documents(data_dir: str) -> List:
    """
    Load all markdown documents from the data directory.
    
    Args:
        data_dir: Path to directory containing markdown files
        
    Returns:
        List of Document objects
    """
    print("📂 Loading documents from:", data_dir)
    
    # DirectoryLoader automatically loads all files matching the pattern
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.md",  # Load all .md files
        loader_cls=TextLoader,
        show_progress=True
    )
    
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} document(s)\n")
    
    return documents


def split_documents(documents: List, chunk_size: int, chunk_overlap: int) -> List:
    """
    Split documents into smaller chunks for better retrieval.
    
    Why chunking matters:
    - Embeddings capture meaning better for focused text segments
    - Smaller chunks = more precise retrieval
    - Prevents overwhelming the LLM with too much context
    
    Args:
        documents: List of Document objects
        chunk_size: Target size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunked Document objects
    """
    print(f"✂️  Splitting documents into chunks...")
    print(f"   Chunk size: {chunk_size} characters")
    print(f"   Chunk overlap: {chunk_overlap} characters\n")
    
    # RecursiveCharacterTextSplitter tries to split on natural boundaries
    # (paragraphs, sentences) rather than arbitrary character positions
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # Try these separators in order
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks from {len(documents)} documents\n")
    
    return chunks


def create_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize the embedding model.
    
    This model converts text into 384-dimensional vectors.
    The model is downloaded and cached locally on first run.
    
    Returns:
        HuggingFaceEmbeddings instance
    """
    print(f"🧠 Initializing embedding model: {EMBEDDING_MODEL}")
    print("   (First run will download the model - ~80MB)\n")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have a GPU
        encode_kwargs={'normalize_embeddings': True}  # Normalize for cosine similarity
    )
    
    return embeddings


def create_vectorstore(chunks: List, embeddings: HuggingFaceEmbeddings, 
                      persist_dir: str) -> Chroma:
    """
    Create embeddings for all chunks and store in ChromaDB.
    
    What happens here:
    1. Each text chunk is converted to a vector (embedding)
    2. Vectors are stored in ChromaDB with metadata
    3. ChromaDB creates an index for fast similarity search
    
    Args:
        chunks: List of document chunks
        embeddings: Embedding model
        persist_dir: Directory to save the vector database
        
    Returns:
        Chroma vectorstore instance
    """
    print(f"💾 Creating vector store in: {persist_dir}")
    print(f"   Processing {len(chunks)} chunks...\n")
    
    # Remove existing vectorstore if it exists
    if os.path.exists(persist_dir):
        import shutil
        shutil.rmtree(persist_dir)
        print("   Removed existing vector store\n")
    
    # Create vectorstore with progress bar
    # This is where the "magic" happens - text becomes searchable vectors!
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="space_exploration"
    )
    
    print("✅ Vector store created successfully!\n")
    
    return vectorstore


def main():
    """Main ingestion pipeline."""
    print_header("RAG System - Document Ingestion")
    
    print("This script will:")
    print("1. Load markdown documents")
    print("2. Split them into chunks")
    print("3. Create embeddings (vector representations)")
    print("4. Store in ChromaDB vector database\n")
    print_separator()
    print()
    
    # Get absolute paths
    script_dir = Path(__file__).parent
    data_dir = (script_dir / DATA_DIR).resolve()
    vectorstore_dir = (script_dir / VECTORSTORE_DIR).resolve()
    
    # Step 1: Load documents
    documents = load_documents(str(data_dir))
    
    if not documents:
        print("❌ No documents found! Please check the data directory.")
        return
    
    # Step 2: Split into chunks
    chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    
    # Step 3: Initialize embedding model
    embeddings = create_embeddings()
    
    # Step 4: Create vector store
    vectorstore = create_vectorstore(chunks, embeddings, str(vectorstore_dir))
    
    # Display statistics
    display_rag_stats(vectorstore)
    
    print_separator()
    print("✨ Ingestion complete! You can now run query.py to ask questions.\n")


if __name__ == "__main__":
    main()
