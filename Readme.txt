# Bengali PDF RAG System

A Retrieval-Augmented Generation (RAG) system for Bengali PDF documents using FastAPI, Streamlit, and Groq LLM.

## Setup Guide

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- Groq API key

### Quick Start
1. Clone the repository:
```bash
git clone https://github.com/PromitSarker/10ms.git
cd 10ms
```

2. Create `.env` file:
```bash
GROQ_API_KEY=your_api_key_here
```

3. Run with Docker:
```bash
docker-compose up --build
```

Access the application:
- Web UI: http://127.0.0.1:8501
- API: http://127.0.0.1:8000

## 🛠️ Tools & Technologies

### Core Components
- **FastAPI**: Backend API framework
- **Streamlit**: Interactive web interface
- **ChromaDB**: Vector database for embeddings
- **Groq**: LLM for question answering
- **LayoutParser**: PDF layout analysis
- **Tesseract OCR**: Bengali text extraction
- **Bengali-BERT**: Text correction
- **Bengali Sentence BERT**: Text embeddings

### Key Libraries
- `bangla-pdf-ocr`: OCR processing solely for Bengali
- `PyMuPDF`: PDF manipulation
- `opencv-python`: Image preprocessing
- `sentence-transformers`: Text embeddings
- `transformers`: BERT-based correction
- `layoutparser`: Document layout analysis

##  Sample Queries & Outputs

### Bengali Queries
```bengali
Q: অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
A: অনুপমের ভাষায় সুপুরুষ হলো শম্ভুনাথ।

Q: কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
A: অনুপমের ভাগ্য দেবতা হলেন তার মামা।
```

## API Documentation

### Endpoints

#### 1. Process PDF
```http
POST /process
Content-Type: multipart/form-data

file: <pdf_file>
```

#### 2. Chat with Documents
```http
POST /chat
Content-Type: application/json

{
    "question": "string",
    "top_k": int
}
```

## Technical Deep Dive

### Text Extraction Strategy
I've used a multi-layered approach:
1. Primary: `pdfplumber` for initial text extraction
2. Fallback: Custom OCR pipeline using:
   - LayoutParser for structure analysis
   - Image preprocessing with OpenCV
   - Tesseract OCR with Bengali support
   - BERT-based text correction

Challenges faced:
- Complex Bengali font rendering
- PDF that was not so easy to extract using OCR. As a result couple of normal libraries like PymuPDF and other fails.
- Table and figure handling
- Document layout preservation

### Chunking Strategy
We implement a hybrid chunking approach:
```python
def chunk_bengali(text: str, max_len: int = 600) -> List[str]:
    # Block-level splitting first
    blocks = re.split(r"\n\s*\n", text.strip())
    # Then sentence-level when needed
```

This works well because:
- Preserves paragraph context
- Respects natural language boundaries
- Handles Bengali sentence markers (।)
- Maintains semantic coherence
- Doesn't get lost in contexts

### Embedding Model
I've used `l3cube-pune/bengali-sentence-bert-nli` because:
- This particular thing was specifically trained on Bengali text
- Fine-tuned for semantic similarity
- Handles Bengali Input quite good 
- Good performance on natural language inference

### Similarity Search & Storage
- **Vector DB**: Used ChromaDB with cosine similarity.
- **Storage**: Persistent disk storage for embeddings
- **Retrieval**: Top-k semantic search
- **Chunking**: Hybrid paragraph/sentence approach

### Query-Document Matching
To ensure meaningful comparisons:
1. Query preprocessing
2. Context window selection
3. Semantic similarity ranking
4. Response generation with context

For vague queries:
- Use larger context windows
- Implement query expansion
- Return multiple relevant chunks
- Based on context tries to stay relevant without spitting weird answers

### Results & Improvements
Current performance metrics:
- Accurate layout detection: 
- OCR accuracy: 
- Relevant chunk retrieval: 

Potential improvements:
1. Better Bengali-specific preprocessing can be done if try Adaptive Binarization, add an Image quality check and enhance bengali text cleaning. Text cleaning enhancing might improve the detection and chunk clarity more. In that way the LLM can have more clear context.
2. More sophisticated chunking
3. Query reformulation
4. Hybrid retrieval methods

## Project Structure
```bash
.
├── app.py              # Streamlit interface
├── main.py            # FastAPI backend
├── requirements.txt   # Dependencies
├── Dockerfile        # Container configuration
├── docker-compose.yml # Multi-container setup
├── .env              # Environment variables
├── chroma_db/        # Vector database storage
└── chunk_store.json  # Text chunk storage
```

## References & Attribution

- Bengali BERT: [sagorsarker/bangla-bert-base](https://huggingface.co/sagorsarker/bangla-bert-base)
- Sentence BERT: [l3cube-pune/bengali-sentence-bert-nli](https://huggingface.co/l3cube-pune/bengali-sentence-bert-nli)
- Layout Detection: [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)

