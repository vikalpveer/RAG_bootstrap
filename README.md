# Space Exploration RAG System 🚀

A self-contained, educational RAG (Retrieval-Augmented Generation) project to understand embeddings, vector stores, and how LLMs can reason over custom knowledge bases.

## 📚 What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that enhances LLM responses by:

1. **Retrieval**: Finding relevant information from a knowledge base
2. **Augmentation**: Adding that information to the LLM prompt
3. **Generation**: Having the LLM generate an answer based on the retrieved context

This allows LLMs to answer questions about specific information they weren't trained on!

## 🎯 Learning Objectives

This project helps you understand:

- **Embeddings**: How text is converted to numerical vectors
- **Vector Stores**: How similar documents are found using vector similarity
- **Chunking**: Breaking documents into optimal sizes for retrieval
- **Semantic Search**: Finding relevant information by meaning, not just keywords
- **RAG Pipeline**: The complete flow from documents to answers

## 📁 Project Structure

```
RAG_Poc/
├── data/                          # Knowledge base (5 markdown files)
│   ├── apollo_missions.md         # Apollo program history
│   ├── astronauts.md              # Famous astronauts
│   ├── future_missions.md         # Upcoming space missions
│   ├── planets.md                 # Solar system planets
│   └── space_technology.md        # Rockets and spacecraft
├── src/                           # Python scripts
│   ├── ingest.py                  # Document ingestion & embedding creation
│   ├── query.py                   # RAG query system
│   └── utils.py                   # Helper functions
├── vectorstore/                   # ChromaDB storage (auto-created)
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## 🛠️ Technology Stack

- **Python 3.8+**: Programming language
- **LangChain**: RAG orchestration framework
- **ChromaDB**: Local vector database (no server needed)
- **sentence-transformers**: Creates embeddings from text
- **Ollama**: Runs LLMs locally (llama2, mistral, etc.)

## 🚀 Setup Instructions

### 1. Prerequisites

**Install Ollama** (for running local LLMs):
```bash
# macOS
brew install ollama

# Or download from: https://ollama.ai
```

**Pull an LLM model**:
```bash
ollama pull llama2
# Or try: ollama pull mistral, ollama pull llama3, etc.
```

**Start Ollama** (if not already running):
```bash
ollama serve
```

### 2. Python Environment

**Create a virtual environment**:
```bash
cd RAG_Poc
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Install dependencies**:
```bash
pip install -r requirements.txt
```

> **Note**: First run will download the embedding model (~80MB) and may take a few minutes.

### 3. Ingest Documents

This step creates embeddings and builds the vector database:

```bash
cd src
python ingest.py
```

**What happens**:
1. Loads 5 markdown files from `data/`
2. Splits them into ~1000 character chunks
3. Creates embeddings (384-dimensional vectors) for each chunk
4. Stores embeddings in ChromaDB (`vectorstore/` directory)

**Expected output**:
```
📂 Loading documents...
✅ Loaded 5 document(s)
✂️  Splitting documents into chunks...
✅ Created ~150 chunks
🧠 Initializing embedding model...
💾 Creating vector store...
✅ Vector store created successfully!
```

### 4. Query the System

Run the query interface:

```bash
python query.py
```

**Choose a mode**:
- **Option 1**: Run example queries (demonstrates the system)
- **Option 2**: Interactive mode (ask your own questions)

## 💡 Example Queries

Try asking questions like:

- "Who was the first person to walk on the Moon?"
- "What is the Artemis program?"
- "Tell me about the Falcon 9 rocket."
- "Which planet has the strongest winds?"
- "What are the goals of the Mars Sample Return mission?"
- "How many moons does Jupiter have?"
- "What happened during Apollo 13?"

## 🔍 How It Works

### Document Ingestion (`ingest.py`)

```
Markdown Files → Load → Split into Chunks → Create Embeddings → Store in ChromaDB
```

**Key concepts**:
- **Chunking**: Documents are split into ~1000 character pieces with 200 character overlap
- **Embeddings**: Each chunk becomes a 384-dimensional vector that captures its meaning
- **Vector Store**: ChromaDB indexes these vectors for fast similarity search

### Query Process (`query.py`)

```
User Question → Create Embedding → Find Similar Chunks → Add to Prompt → LLM Generates Answer
```

**The RAG flow**:
1. **Your question** is converted to an embedding (same model as ingestion)
2. **Vector search** finds the 3 most similar document chunks
3. **Context + Question** are combined in a prompt
4. **LLM generates** an answer based on the retrieved context
5. **Sources are shown** so you can verify the information

## 🎓 Understanding the Code

### Key Files Explained

**`ingest.py`** - Document Ingestion
- Loads markdown files
- Splits into chunks (RecursiveCharacterTextSplitter)
- Creates embeddings (sentence-transformers)
- Stores in ChromaDB

**`query.py`** - RAG Query System
- Loads vector store
- Converts queries to embeddings
- Retrieves relevant chunks (similarity search)
- Sends context + query to LLM
- Displays answer with sources

**`utils.py`** - Helper Functions
- Pretty printing
- Document formatting
- Statistics display

### Important Parameters

**In `ingest.py`**:
```python
CHUNK_SIZE = 1000        # Characters per chunk
CHUNK_OVERLAP = 200      # Overlap to maintain context
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

**In `query.py`**:
```python
LLM_MODEL = "llama2"     # Ollama model to use
TOP_K_RESULTS = 3        # Number of chunks to retrieve
```

## 🔧 Customization

### Use a Different LLM Model

Edit `src/query.py`:
```python
LLM_MODEL = "mistral"  # or "llama3", "codellama", etc.
```

Then pull the model:
```bash
ollama pull mistral
```

### Change Chunk Size

Edit `src/ingest.py`:
```python
CHUNK_SIZE = 500         # Smaller chunks = more precise retrieval
CHUNK_OVERLAP = 100      # Adjust overlap accordingly
```

Then re-run ingestion:
```bash
python ingest.py
```

### Add Your Own Documents

1. Add markdown files to `data/` directory
2. Re-run ingestion: `python ingest.py`
3. Query your new knowledge base!

### Retrieve More Context

Edit `src/query.py`:
```python
TOP_K_RESULTS = 5  # Retrieve 5 chunks instead of 3
```

## 🧪 Experimentation Ideas

1. **Compare embedding models**: Try different models from sentence-transformers
2. **Test chunk sizes**: See how chunk size affects retrieval quality
3. **Try different LLMs**: Compare llama2, mistral, llama3 responses
4. **Add more documents**: Expand the knowledge base
5. **Modify prompts**: Experiment with different prompt templates
6. **Implement hybrid search**: Combine semantic + keyword search
7. **Add re-ranking**: Use a re-ranker model to improve retrieval

## 📊 Understanding Embeddings

Embeddings convert text to vectors in high-dimensional space:

```
"Apollo 11 landed on the Moon" → [0.23, -0.45, 0.67, ..., 0.12]  (384 numbers)
"First Moon landing mission"   → [0.21, -0.43, 0.69, ..., 0.15]  (similar!)
"Jupiter is a gas giant"       → [-0.67, 0.23, -0.12, ..., 0.89] (different)
```

**Similar meanings = similar vectors** → enables semantic search!

## 🐛 Troubleshooting

### "Ollama not found" or connection errors
```bash
# Make sure Ollama is installed and running
ollama serve

# In another terminal, verify it works:
ollama list
```

### "Vector store not found"
```bash
# Run ingestion first:
cd src
python ingest.py
```

### "Model not found"
```bash
# Pull the model:
ollama pull llama2
```

### Slow performance
- First run downloads embedding model (~80MB)
- Subsequent runs are much faster
- Consider using a GPU for faster embeddings (change `device='cpu'` to `device='cuda'`)

## 📖 Additional Resources

- **LangChain Docs**: https://python.langchain.com/docs/
- **ChromaDB Docs**: https://docs.trychroma.com/
- **Ollama Models**: https://ollama.ai/library
- **Sentence Transformers**: https://www.sbert.net/
- **RAG Paper**: https://arxiv.org/abs/2005.11401

## 🎯 Next Steps

After mastering this project, explore:

1. **Advanced RAG techniques**:
   - Hybrid search (semantic + keyword)
   - Re-ranking retrieved documents
   - Query expansion and refinement
   - Multi-query retrieval

2. **Production considerations**:
   - API endpoints (FastAPI)
   - Caching strategies
   - Monitoring and logging
   - Scaling vector stores

3. **Alternative tools**:
   - Pinecone, Weaviate, Qdrant (cloud vector DBs)
   - OpenAI embeddings (ada-002)
   - LlamaIndex framework

## 📝 License

This is an educational project. Feel free to use and modify for learning purposes!

## 🤝 Contributing

This is a learning project, but suggestions for improvements are welcome!

---

**Happy Learning! 🚀**

*Built with ❤️ to understand RAG, embeddings, and vector stores*
