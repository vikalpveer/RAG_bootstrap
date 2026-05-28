"""
RAG Query System

This script demonstrates how to query a RAG system:
1. Load the vector database
2. Convert user query to embedding
3. Find similar document chunks (retrieval)
4. Send context + query to LLM (augmentation + generation)

This is where RAG shows its power: answering questions based on your custom knowledge base!
"""

import os
import sys
from pathlib import Path
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate

from utils import (
    print_header, 
    print_separator, 
    print_documents,
    display_rag_stats
)


# Configuration
VECTORSTORE_DIR = "../vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama2"  # Change to your preferred Ollama model
TOP_K_RESULTS = 3  # Number of document chunks to retrieve


def load_vectorstore(persist_dir: str, embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Load the existing vector database.
    
    Args:
        persist_dir: Directory where vectorstore is saved
        embeddings: Embedding model (must be same as used in ingestion)
        
    Returns:
        Chroma vectorstore instance
    """
    print(f"📂 Loading vector store from: {persist_dir}\n")
    
    if not os.path.exists(persist_dir):
        print("❌ Vector store not found! Please run ingest.py first.")
        sys.exit(1)
    
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="space_exploration"
    )
    
    print("✅ Vector store loaded successfully!\n")
    return vectorstore


def initialize_llm(model_name: str) -> Ollama:
    """
    Initialize the local LLM using Ollama.
    
    Make sure you have Ollama installed and the model pulled:
    - Install: https://ollama.ai
    - Pull model: ollama pull llama2
    
    Args:
        model_name: Name of the Ollama model
        
    Returns:
        Ollama LLM instance
    """
    print(f"🤖 Initializing LLM: {model_name}")
    print("   (Make sure Ollama is running and model is pulled)\n")
    
    llm = Ollama(
        model=model_name,
        callbacks=[StreamingStdOutCallbackHandler()],  # Stream responses
        temperature=0.7,  # Balance between creativity and consistency
    )
    
    return llm


def create_rag_chain(vectorstore: Chroma, llm: Ollama, k: int = 3):
    """
    Create a RAG chain that combines retrieval and generation.
    
    The RAG process:
    1. Query → Embedding
    2. Find top K similar chunks in vectorstore (Retrieval)
    3. Combine chunks with query in prompt (Augmentation)
    4. Send to LLM for answer (Generation)
    
    Args:
        vectorstore: Vector database
        llm: Language model
        k: Number of documents to retrieve
        
    Returns:
        RetrievalQA chain
    """
    # Create a retriever from the vectorstore
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # Use cosine similarity
        search_kwargs={"k": k}  # Return top k results
    )
    
    # Custom prompt template for better responses
    prompt_template = """You are a knowledgeable assistant specializing in space exploration. 
Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know - don't make up information.
Provide detailed, accurate answers based on the context provided.

Context:
{context}

Question: {question}

Detailed Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create the RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" means put all context in one prompt
        retriever=retriever,
        return_source_documents=True,  # Return the source chunks
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain


def query_rag_system(qa_chain, query: str, show_sources: bool = True):
    """
    Query the RAG system and display results.
    
    Args:
        qa_chain: RetrievalQA chain
        query: User's question
        show_sources: Whether to display source documents
    """
    print_separator("=")
    print(f"❓ Question: {query}\n")
    print_separator("-")
    print("🤔 Thinking...\n")
    
    # Execute the query
    result = qa_chain({"query": query})
    
    print("\n")
    print_separator("-")
    print("💡 Answer:")
    print(result['result'])
    print()
    
    # Show source documents if requested
    if show_sources and 'source_documents' in result:
        print_separator("-")
        print_documents(result['source_documents'])
    
    print_separator("=")
    print()


def run_example_queries(qa_chain):
    """Run some example queries to demonstrate the system."""
    example_queries = [
        "Who was the first person to walk on the Moon?",
        "What is the Artemis program?",
        "Tell me about the Falcon 9 rocket.",
        "Which planet has the strongest winds?",
        "What are the goals of the Mars Sample Return mission?"
    ]
    
    print_header("Example Queries")
    print("Running example queries to demonstrate the RAG system...\n")
    
    for i, query in enumerate(example_queries, 1):
        print(f"\n{'='*80}")
        print(f"Example {i}/{len(example_queries)}")
        query_rag_system(qa_chain, query, show_sources=True)
        
        if i < len(example_queries):
            input("Press Enter to continue to next example...")


def interactive_mode(qa_chain):
    """Run in interactive mode for custom queries."""
    print_header("Interactive Mode")
    print("Ask questions about space exploration!")
    print("Type 'quit' or 'exit' to stop.\n")
    print_separator()
    print()
    
    while True:
        try:
            query = input("Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print()
            query_rag_system(qa_chain, query, show_sources=True)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    """Main query pipeline."""
    print_header("RAG System - Query Interface")
    
    # Get absolute paths
    script_dir = Path(__file__).parent
    vectorstore_dir = (script_dir / VECTORSTORE_DIR).resolve()
    
    # Initialize embeddings (must match ingestion)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Load vectorstore
    vectorstore = load_vectorstore(str(vectorstore_dir), embeddings)
    display_rag_stats(vectorstore)
    print()
    
    # Initialize LLM
    try:
        llm = initialize_llm(LLM_MODEL)
    except Exception as e:
        print(f"\n❌ Error initializing LLM: {e}")
        print("\nMake sure:")
        print("1. Ollama is installed: https://ollama.ai")
        print(f"2. Model is pulled: ollama pull {LLM_MODEL}")
        print("3. Ollama service is running")
        sys.exit(1)
    
    # Create RAG chain
    print("🔗 Creating RAG chain...\n")
    qa_chain = create_rag_chain(vectorstore, llm, k=TOP_K_RESULTS)
    print("✅ RAG system ready!\n")
    print_separator()
    
    # Choose mode
    print("\nChoose mode:")
    print("1. Run example queries")
    print("2. Interactive mode (ask your own questions)")
    print()
    
    choice = input("Enter choice (1 or 2): ").strip()
    print()
    
    if choice == "1":
        run_example_queries(qa_chain)
    else:
        interactive_mode(qa_chain)


if __name__ == "__main__":
    main()
