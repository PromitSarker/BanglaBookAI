# ---------- Imports ----------
import re
import io
import os
import cv2
import pdfplumber
import numpy as np
import unicodedata
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import uvicorn
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import groq
from dotenv import load_dotenv
import regex as re2  # Unicode script support

load_dotenv()
app = FastAPI()

# Set this to your custom tessdata directory if using a custom ben.traineddata, else None
TESSDATA_DIR = None  # e.g. "/home/user/tessdata" or None for default

# ---------- Bengali SBERT ----------
model = SentenceTransformer("l3cube-pune/bengali-sentence-bert-nli")

# ---------- ChromaDB ----------
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)
collection = chroma_client.get_or_create_collection(
    name="bengali_docs",
    metadata={"hnsw:space": "cosine"}
)

# ---------- Groq ----------
groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------- Text Cleaners ----------
def clean_bengali(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re2.sub(r"[\u00AD\u200C\u200D]", "", text)  # remove soft-hyphen, ZWJ/ZWNJ
    text = re2.sub(r"\s+", " ", text)
    text = re2.sub(r"[^\p{Script=Bengali}\p{P}\p{N}\s]", "", text)
    return text.strip()

COMMON_FIXES = {
    "ত‌্য": "ত্য",
    "্ে": "ে",
    "ি্": "ি",
    "র্": "র্",  # placeholder, can extend with more
}

def post_correct(text: str) -> str:
    for wrong, right in COMMON_FIXES.items():
        text = text.replace(wrong, right)
    return text

def join_broken_words(text: str) -> str:
    # Join broken parts if single space splits consonants
    return re.sub(
        r'(\S)(\s+)(\S)',
        lambda m: m.group(1) + m.group(3) if len(m.group(2)) == 1 else m.group(0),
        text
    )

# ---------- OCR Utilities ----------
def render_page_highres(page: fitz.Page, dpi: int = 500) -> bytes:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")

def preprocess_image_advanced(img_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)

    # Upscale for better OCR
    h, w = img_np.shape[:2]
    img_np = cv2.resize(img_np, (int(w * 2), int(h * 2)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Denoise using Non-local Means
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # Adaptive threshold
    bin_img = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15
    )

    # Morph close to connect characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    return Image.fromarray(closed)

def ocr_page(img_bytes: bytes) -> str:
    processed_img = preprocess_image_advanced(img_bytes)
    config_parts = ["--oem 1", "--psm 6", "-l ben+eng+osd", "--load_system_dawg 0", "--load_freq_dawg 0"]
    if TESSDATA_DIR:
        config_parts.insert(0, f"--tessdata-dir {TESSDATA_DIR}")
    config = " ".join(config_parts)

    raw = pytesseract.image_to_string(processed_img, config=config)
    cleaned = clean_bengali(raw)
    joined = join_broken_words(cleaned)
    corrected = post_correct(joined)
    return corrected

def chunk_bengali(text: str, max_len: int = 600) -> List[str]:
    sentences = re2.split(r'(?<=[।!?])\s*', text)
    chunks, chunk = [], ""
    for sent in sentences:
        if len(chunk) + len(sent) < max_len:
            chunk += " " + sent
        else:
            if chunk:
                chunks.append(chunk.strip())
            chunk = sent
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def extract_text_pdfplumber(pdf_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages)

# ---------- /process ----------
@app.post("/process")
async def process_pdf(file: UploadFile = File(...)):
    pdf_bytes = await file.read()

    # Try extracting text first
    text = extract_text_pdfplumber(pdf_bytes)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_text = []

    for page in doc:
        raw = page.get_text("text")
        if raw and len(raw.strip()) > 20:
            all_text.append(clean_bengali(raw))
        else:
            img_bytes = render_page_highres(page)
            all_text.append(ocr_page(img_bytes))
    doc.close()

    full_text = " ".join(all_text)
    chunks = chunk_bengali(full_text)
    embeddings = model.encode(chunks, normalize_embeddings=True).tolist()
    ids = [f"{file.filename}_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)

    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "stored_count": len(chunks)
    }

# ---------- /chat ----------
class ChatRequest(BaseModel):
    question: str
    top_k: int = 5

@app.post("/chat")
async def chat_with_docs(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question.")

    q_emb = model.encode([req.question], normalize_embeddings=True).tolist()
    hits = collection.query(query_embeddings=q_emb, n_results=req.top_k)
    contexts = hits["documents"][0]
    if not contexts or not contexts[0].strip():
        return {"answer": "প্রাসঙ্গিক তথ্য পাওয়া যায়নি।"}

    context_str = "\n\n---\n\n".join(contexts)
    prompt = (
        "তুমি একজন বাংলা ভাষী সহকারী। নিচের প্রসঙ্গ ব্যবহার করে **শুধু বাংলায়** উত্তর দাও।\n\n"
        "Do not use your own reasoning, and do not give any explanation"
        f"প্রসঙ্গ:\n{context_str}\n\n"
        f"প্রশ্ন: {req.question}\nউত্তর:"
    )

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.15
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"answer": answer}

# ---------- run ----------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
