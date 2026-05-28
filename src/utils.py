"""
Utility functions for the RAG system.
This module contains helper functions for document processing and display.
"""

from typing import List, Dict
from langchain.schema import Document


def print_separator(char: str = "=", length: int = 80) -> None:
    """Print a separator line for better output formatting."""
    print(char * length)


def print_documents(documents: List[Document], max_content_length: int = 200) -> None:
    """
    Pretty print retrieved documents with metadata.
    
    Args:
        documents: List of LangChain Document objects
        max_content_length: Maximum characters to display from content
    """
    print(f"\n📚 Retrieved {len(documents)} relevant document(s):\n")
    
    for i, doc in enumerate(documents, 1):
        print(f"Document {i}:")
        print(f"  Source: {doc.metadata.get('source', 'Unknown')}")
        
        # Display content preview
        content = doc.page_content
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        print(f"  Content: {content}")
        print()


def format_prompt_with_context(query: str, context_docs: List[Document]) -> str:
    """
    Format a prompt with retrieved context for the LLM.
    
    This is a key part of RAG: combining retrieved documents with the user's query.
    
    Args:
        query: User's question
        context_docs: Retrieved relevant documents
        
    Returns:
        Formatted prompt string
    """
    # Combine all document contents
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    # Create the prompt template
    prompt = f"""You are a knowledgeable assistant specializing in space exploration. 
Use the following context to answer the question. If you cannot answer the question based on the context, say so.

Context:
{context}

Question: {query}

Answer: """
    
    return prompt


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Chunking is important in RAG because:
    1. Embeddings work better on smaller, focused text segments
    2. Retrieval is more precise when matching specific information
    3. LLMs have context length limits
    
    Args:
        text: Text to chunk
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap
        
    return chunks


def display_rag_stats(vectorstore) -> None:
    """
    Display statistics about the vector store.
    
    Args:
        vectorstore: ChromaDB vectorstore instance
    """
    try:
        collection = vectorstore._collection
        count = collection.count()
        print(f"\n📊 Vector Store Statistics:")
        print(f"  Total document chunks: {count}")
        print(f"  Collection name: {collection.name}")
    except Exception as e:
        print(f"Could not retrieve stats: {e}")


def print_header(text: str) -> None:
    """Print a formatted header."""
    print_separator("=")
    print(f"  {text}")
    print_separator("=")
    print()
