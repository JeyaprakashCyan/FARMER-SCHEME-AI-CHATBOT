# 🌾 Farmer Scheme Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about Indian farmer and agricultural government schemes using a curated collection of government scheme documents.

The system uses PDF extraction, OCR, metadata-aware document processing, semantic embeddings, FAISS vector search, query rewriting, scheme-aware routing, reranking, fact-aware retrieval, conversation memory, and grounded LLM answer generation.

---

## 📌 Project Overview

The Farmer Scheme Chatbot allows users to ask natural-language questions about agricultural and farmer welfare schemes.

Instead of relying only on the language model's general knowledge, the chatbot retrieves relevant information from a collection of scheme documents and generates answers based on the retrieved evidence.

### Knowledge Base

The current knowledge base contains:

- 26 farmer/agriculture scheme PDFs
- 90 document pages
- OCR processing for scanned/image-based pages
- 221 final retrieval chunks
- Structured metadata
- FAISS vector index

---

# 🎯 Objectives

The main objectives of this project are:

- Provide accurate answers about farmer schemes.
- Use official/document-based information instead of relying only on LLM knowledge.
- Support scanned PDFs using OCR.
- Improve retrieval accuracy using metadata and scheme routing.
- Improve relevance using reranking.
- Handle factual and numeric questions accurately.
- Support conversational follow-up questions.
- Provide source and page references.
- Provide observability using LangSmith.

---

# 🚀 Key Features

## 1. PDF Knowledge Base

The chatbot works with a collection of farmer and agricultural scheme documents.

Examples include:

- PM-KISAN
- PM Kisan Maan Dhan Yojana
- Agriculture Infrastructure Fund
- Animal Husbandry Infrastructure Development Fund
- Pradhan Mantri Krishi Sinchai Yojana
- Rashtriya Gokul Mission
- Livestock Insurance
- National Beekeeping and Honey Mission
- National Mission on Natural Farming
- Mission for Aatmanirbharta in Pulses
- Fisheries-related schemes
- Kisan Credit Card
- Tea Development & Promotion Scheme
- and other farmer-related schemes.

---

# 🧠 RAG Architecture

The system consists of two major pipelines:

## Offline Knowledge Base Pipeline

```text
PDF Files
    ↓
PDF Text Extraction
    ↓
OCR for Low-Quality / Scanned Pages
    ↓
Document Standardization
    ↓
Metadata Creation
    ↓
Smart Chunking
    ↓
Embeddings
    ↓
FAISS Vector Store
Online Question Answering Pipeline
User Question
      ↓
Conversation Memory
      ↓
Query Rewriting
      ↓
Scheme Detection
      ↓
Fact / Year Detection
      ↓
FAISS Retrieval
      ↓
Additional Fact Retrieval
      ↓
Deduplication
      ↓
Reranking
      ↓
Scheme / Fact Boosting
      ↓
Relevance Threshold
      ↓
Quality Control
      ↓
Final Context
      ↓
Grounded Answer Generation
      ↓
Source Extraction
      ↓
Streamlit UI
🔎 Retrieval Architecture

The retrieval system uses multiple stages.

Stage 1 — Semantic Retrieval

The user's question is converted into an embedding and compared against the FAISS vector store.

Stage 2 — Scheme Routing

The system identifies the farmer scheme mentioned in the question.

Example:

Question:
What is PM-KISAN?

Detected Scheme:
Pradhan Mantri Kisan Samman Nidhi

Target Document:
Pradhan Mantri Kisan Samman Nidhi.pdf
Stage 3 — Fact / Year Retrieval

Fact-oriented questions receive additional retrieval queries.

For example:

How much livestock insured in financial year (2024-25)?

The system generates additional retrieval queries related to:

total number
insured livestock
coverage
beneficiaries
financial year
Stage 4 — Reranking

Retrieved documents are reranked using:

BAAI/bge-reranker-base
Stage 5 — Relevance Filtering

Low-relevance documents are removed before the final context is sent to the LLM.

🧩 Conversation Memory

The chatbot supports follow-up questions.

Example:

User:
What is PM-KISAN?

Assistant:
PM-KISAN is an income support scheme...

User:
What are its benefits?

Assistant:
The benefits of PM-KISAN include...

The second question can be rewritten into a standalone question using conversation history.

🔄 Query Rewriting

Context-dependent questions are rewritten before retrieval.

Example:

Previous question:
What is PM-KISAN?

Follow-up:
What are its benefits?

Rewritten query:
What are the benefits of PM-KISAN?

This improves retrieval for conversational questions.

📊 Fact-Aware Answer Generation

The answer generation layer is designed to handle exact factual questions.

For example:

Question:
How much livestock insured in financial year (2024-25)?

Retrieved context:

21.01 Lakh
Livestock insured in financial year (2024-25)

Answer:

21.01 Lakh livestock were insured in financial year (2024-25).

The answer generator is instructed to match:

VALUE + ENTITY + TIME PERIOD

This prevents unrelated numbers from being incorrectly combined.

📚 Source Attribution

Answers include source information whenever available.

Example:

Answer:
21.01 Lakh livestock were insured in financial year (2024-25).

Source:
Livestock Insurance - An introduction.pdf
Page 1

This improves transparency and allows users to verify the information.

🛠️ Technology Stack
Programming Language
Python
Frameworks / Libraries
LangChain
LangChain OpenAI
FAISS
Sentence Transformers
Streamlit
PyMuPDF / PDF processing tools
OCR tools
LangSmith
LLM
gpt-4o-mini
Embedding Model
BAAI/bge-small-en-v1.5
Reranker
BAAI/bge-reranker-base
Vector Database
FAISS
📂 Project Structure
farmer-scheme-chatbot/
│
├── README.md
├── requirements.txt
├── .env
├── .gitignore
│
├── data/
│   ├── pdf/
│   │   ├── Agriculture Infrastructure Fund.pdf
│   │   ├── Animal Husbandry Infrastructure Development Fund.pdf
│   │   ├── ...
│   │   └── Vibrant Villages Programme.pdf
│   │
│   └── vector_store/
│       └── faiss_index/
│
└── src/
    └── rag/
        ├── app.py
        ├── query_pipeline.py
        ├── answer_generator.py
        ├── conversation_memory.py
        ├── query_rewriter.py
        ├── scheme_router.py
        ├── retriever.py
        ├── reranker.py
        ├── embeddings.py
        ├── vector_store.py
        ├── chunking.py
        ├── metadata_schema.py
        └── ...
📄 Important Python Modules
File	Responsibility
app.py	Streamlit user interface
query_pipeline.py	Main RAG orchestration
answer_generator.py	Grounded LLM answer generation
conversation_memory.py	Conversation/session memory
query_rewriter.py	Follow-up question rewriting
scheme_router.py	Scheme detection and routing
retriever.py	Vector retrieval and filtering
reranker.py	Cross-encoder reranking
embeddings.py	Embedding model
vector_store.py	FAISS creation/loading/search
chunking.py	Document chunking
metadata_schema.py	Metadata structure and validation
⚙️ Installation

Clone the repository:

git clone <repository-url>

Move into the project:

cd farmer-scheme-chatbot

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
🔐 Environment Variables

Create a .env file in the project root.

Example:

OPENAI_API_KEY=your_openai_api_key

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=farmer-scheme-chatbot

Never commit .env to GitHub.

🏗️ Knowledge Base Creation

The knowledge-base pipeline performs:

PDF Loading
    ↓
OCR
    ↓
Document Creation
    ↓
Metadata
    ↓
Chunking
    ↓
Embeddings
    ↓
FAISS

The resulting vector database is stored under:

data/vector_store/faiss_index/
▶️ Running the Chatbot

Start the Streamlit application:

streamlit run app.py

If the application is located inside src/rag:

streamlit run src/rag/app.py

The application will open in the browser.

💬 Example Questions
General Scheme Questions
What is PM-KISAN?
What are the benefits of PM-KISAN?
Who is eligible for PM-KISAN?
Fact Questions
How much livestock insured in financial year (2024-25)?
What percentage subsidy is available?
How many beneficiaries are covered?
Conversational Questions
What is PM-KISAN?
What are its benefits?
Who is eligible for it?
🔬 LangSmith Observability

LangSmith is used to monitor and debug the RAG pipeline.

The tracing architecture is:

User Question
      ↓
Query Pipeline
      ↓
Retrieval
      ↓
Reranking
      ↓
Context
      ↓
LLM
      ↓
Answer

LangSmith can be used to inspect:

LLM calls
Retrieval behavior
Reranking
Prompts
Responses
Latency
Errors
End-to-end traces
🎯 Design Goals

The system prioritizes:

Accuracy
Grounded answers
Relevant retrieval
Source transparency
Conversational context
Exact factual extraction
Low hallucination
Explainable retrieval behavior
🔮 Future Improvements

Potential future improvements include:

Hybrid search
BM25 + vector retrieval
Advanced metadata filtering
Better OCR table reconstruction
Multi-language support
Hindi / regional-language support
Citation highlighting
Source document preview
Retrieval evaluation dataset
RAG evaluation metrics
Automated evaluation using LangSmith
Improved query classification
More advanced reranking
Production database/vector-store deployment
Authentication and user management
📈 Current RAG Pipeline
26 PDFs
   ↓
OCR / PDF Extraction
   ↓
90 Pages
   ↓
Document Standardization
   ↓
Metadata
   ↓
Smart Chunking
   ↓
221 Chunks
   ↓
BGE Embeddings
   ↓
FAISS
   ↓
Query
   ↓
Query Rewriting
   ↓
Scheme Routing
   ↓
Fact/Year Retrieval
   ↓
Reranking
   ↓
Relevance Filtering
   ↓
Grounded LLM
   ↓
Answer + Sources
   ↓
Streamlit
   ↓
LangSmith
👨‍💻 Project Status

Current implementation includes:

 PDF ingestion
 OCR processing
 OCR validation
 Document standardization
 Metadata architecture
 Smart chunking
 Embeddings
 FAISS vector store
 Retriever
 Metadata-aware retrieval
 Reranking
 Query rewriting
 Conversation memory
 Scheme routing
 Fact/year retrieval
 Fact-aware answer generation
 Source attribution
 Streamlit UI
 LangSmith tracing
📜 License

This project is intended for educational and research purposes.

🙏 Acknowledgements

This project uses open-source technologies including LangChain, FAISS, Sentence Transformers, Streamlit, and related NLP/OCR tools.


---

## 5. Very important: `.gitignore`

Since you're going to put this project on GitHub, **don't upload your API keys**.

Create another file in the root:

```text
.gitignore

Put:

# Environment
.env
.venv/
venv/

# Python
__pycache__/
*.py[cod]
*.pyo

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Local vector database
data/vector_store/

# Temporary files
*.tmp
*.temp

# Jupyter
.ipynb_checkpoints/
One thing to decide

If you want your 26 PDFs and FAISS index to be available when someone clones the GitHub repository, we need to handle those separately because PDFs/vector indexes can make the repository unnecessarily large.

For now, I recommend:

GitHub
│
├── README.md             ✅
├── src/                  ✅
├── requirements.txt      ✅
├── .gitignore            ✅
├── .env                  ❌
├── data/pdf/             ⚠️ depends on size/license
└── data/vector_store/    ❌

README.md should definitely be in the root. It becomes the first documentation page people see when they open your GitHub repository.