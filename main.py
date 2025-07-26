# main.py
# FastAPI + Bengali PDF RAG (OCR v4 + layout masking + Groq LLM)
# pip install fastapi uvicorn python-multipart pymupdf pdfplumber opencv-python
#             pillow bangla-pdf-ocr sentence-transformers chromadb
#             pytesseract transformers torch indic-nlp-library
#             groq layoutparser[layoutmodels]

import io
import json
import os
import pathlib
import re
import tempfile
import shutil
from typing import List

import cv2
import fitz  # PyMuPDF
import numpy as np
import pdfplumber
import regex as re2
import unicodedata as ud
from bangla_pdf_ocr import process_pdf as bpo_process_pdf
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from groq import Groq
import chromadb
import pytesseract
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uvicorn

load_dotenv()

# ---------- BERT corrector (light) ----------
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

_CORRECTOR_NAME = "sagorsarker/bangla-bert-base"
_corrector_tok = AutoTokenizer.from_pretrained(_CORRECTOR_NAME)
_corrector_mdl = AutoModelForMaskedLM.from_pretrained(_CORRECTOR_NAME)
_corrector_mdl.eval()

app = FastAPI()

def _correct_with_bert(text: str, max_mask: int = 5) -> str:
    tokens = _corrector_tok.tokenize(text)
    if len(tokens) < 3:
        return text
    inputs = _corrector_tok(text, return_tensors="pt")
    with torch.no_grad():
        logits = _corrector_mdl(**inputs).logits[0]
    probs = logits.softmax(-1)
    input_ids = inputs["input_ids"][0]
    logp = probs[torch.arange(len(input_ids)), input_ids]
    worst_idx = torch.topk(-logp, k=min(max_mask, len(input_ids))).indices.tolist()
    new_ids = input_ids.clone()
    for idx in worst_idx:
        masked = input_ids.clone()
        masked[idx] = _corrector_tok.mask_token_id
        masked_logits = _corrector_mdl(input_ids=masked.unsqueeze(0)).logits[0, idx]
        best_id = masked_logits.argmax(-1).item()
        new_ids[idx] = best_id
    corrected = _corrector_tok.decode(new_ids, skip_special_tokens=True)
    return corrected

# ---------- Groq client ----------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------- Vector DB ----------
chroma_client = None
collection = None

def initialize_chroma_db():
    """Initialize or reset ChromaDB with error recovery."""
    global chroma_client, collection
    chroma_dir = "./chroma_db"
    try:
        chroma_client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        collection = chroma_client.get_or_create_collection(
            name="bengali_docs",
            metadata={"hnsw:space": "cosine"}
        )
        return chroma_client, collection
    except Exception as e:
        print(f"ChromaDB init failed: {e}. Resetting...")
        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir)
        chroma_client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        collection = chroma_client.create_collection(
            name="bengali_docs",
            metadata={"hnsw:space": "cosine"}
        )
        return chroma_client, collection

# initialise once
chroma_client, collection = initialize_chroma_db()

# ---------- Key-value store ----------
CHUNK_STORE = pathlib.Path("./chunk_store.json")
def _load_store() -> dict[str, str]:
    return json.loads(CHUNK_STORE.read_text(encoding="utf-8")) if CHUNK_STORE.exists() else {}
def _save_store(store: dict[str, str]) -> None:
    CHUNK_STORE.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")

# ---------- Light cleaners ----------
_HIDDEN = re2.compile(r'[\u00AD\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF]+')
_LAT2BEN_DIGITS = str.maketrans('0123456789', '০১২৩৪৫৬৭৮৯')
_HYPHEN_BREAK = re2.compile(r'(\w+)-\s*\n\s*(\w+)')

def _light_clean(text: str) -> str:
    text = ud.normalize("NFC", text)
    text = _HIDDEN.sub("", text)
    text = text.translate(_LAT2BEN_DIGITS)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _strip_junk(text: str) -> str:
    """
    Remove every kind of junk while keeping
    – Bengali letters, digits, punctuation, whitespace, newlines
    – MCQ option markers (ক), (খ), (গ), (ঘ)
    """
    # 1. Delete null bytes and other invisible control chars
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)

    # 2. Keep only allowed Unicode categories
    allowed = re2.compile(r'[^\p{Bengali}\p{Nd}\p{P}\s]')
    text = allowed.sub('', text)

    # 3. Collapse multiple spaces / newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def join_broken_words(text: str) -> str:
    return _HYPHEN_BREAK.sub(r'\1\2', text)

# ---------- Layout-aware OCR ----------
import layoutparser as lp
layout_model = lp.EfficientDetLayoutModel("lp://efficientdet/PubLayNet/tf_efficientdet_d0/config")

def _preprocess_v2(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle += 90
    if abs(angle) > 0.3:
        M = cv2.getRotationMatrix2D((gray.shape[1]//2, gray.shape[0]//2), angle, 1.0)
        gray = cv2.warpAffine(gray, M, gray.shape[::-1],
                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary

def run_bangla_ocr_on_png_bytes(img_bytes: bytes) -> str:
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    layout = layout_model.detect(img_bgr)
    mask = np.ones(img_bgr.shape[:2], dtype=np.uint8) * 255
    for b in layout:
        if b.type in {"Figure"}:
            x1, y1, x2, y2 = map(int, b.coordinates)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
    img_masked = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
        cv2.imwrite(tmp_in.name, _preprocess_v2(img_masked))
        in_path = tmp_in.name
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp_out:
        out_path = tmp_out.name
    try:
        bpo_process_pdf(in_path, out_path)
        with open(out_path, encoding="utf-8") as f:
            raw = f.read()
    finally:
        os.unlink(in_path); os.unlink(out_path)

    if len(raw.strip()) < 40:
        bin_img = _preprocess_v2(img_masked)
        tess_text = pytesseract.image_to_string(
            Image.fromarray(bin_img), lang="ben", config="--psm 6"
        )
        raw = tess_text if tess_text.strip() else raw

    full_page = raw
    cleaned = _light_clean(full_page)
    cleaned = join_broken_words(cleaned)
    cleaned = _correct_with_bert(cleaned)
    return cleaned

# ---------- Chunking ----------
model = SentenceTransformer("l3cube-pune/bengali-sentence-bert-nli")

def chunk_bengali(text: str, max_len: int = 600) -> List[str]:
    blocks = re.split(r"\n\s*\n", text.strip())
    chunks, cur = [], ""
    for blk in blocks:
        if len(cur) + len(blk) + 1 <= max_len:
            cur += "\n" + blk
        else:
            if cur: chunks.append(cur.strip())
            cur = blk
    if cur: chunks.append(cur.strip())
    final = []
    for chk in chunks:
        if len(chk) <= max_len:
            final.append(chk)
        else:
            sentences = re2.split(r"(?<=[।!?॥])\s*", chk)
            part = ""
            for sent in sentences:
                if len(part) + len(sent) < max_len:
                    part += " " + sent
                else:
                    if part: final.append(part.strip())
                    part = sent
            if part: final.append(part.strip())
    return [c for c in final if c]

def render_page_highres(page: fitz.Page) -> bytes:
    mat = fitz.Matrix(3.0, 3.0)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")

# ---------- FastAPI endpoints ----------
@app.post("/process")
async def process_pdf(file: UploadFile = File(...)):
    global chroma_client, collection
    pdf_bytes = await file.read()
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    text = _light_clean(text)
    text = _strip_junk(text)     # remove junk characters / words

    if len(text) < 20:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_texts = []
        for page in doc:
            raw = page.get_text("text")
            if raw and len(raw.strip()) > 20:
                page_texts.append(_light_clean(raw))
            else:
                img_bytes = render_page_highres(page)
                page_texts.append(run_bangla_ocr_on_png_bytes(img_bytes))
        doc.close()
        text = "\n".join(page_texts)

    chunks = chunk_bengali(text)
    embeddings = model.encode(chunks, normalize_embeddings=True).tolist()
    ids = [f"{file.filename}_{i}" for i in range(len(chunks))]

    collection.add(embeddings=embeddings, ids=ids)
    store = _load_store()
    for cid, chk in zip(ids, chunks):
        store[cid] = chk
    _save_store(store)

    return {"chunks": chunks, "stored_count": len(chunks)}

class ChatRequest(BaseModel):
    question: str
    top_k: int = 5

@app.post("/chat")
async def chat_with_docs(req: ChatRequest):
    global chroma_client, collection
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question.")

    q_emb = model.encode([req.question], normalize_embeddings=True).tolist()
    hits = collection.query(
        query_embeddings=q_emb,
        n_results=req.top_k,
        include=["metadatas"]
    )
    store = _load_store()
    contexts = [store[i] for i in hits["ids"][0] if i in store]
    if not contexts:
        return {"answer": "প্রাসঙ্গিক তথ্য পাওয়া যায়নি।"}

    context_str = "\n\n---\n\n".join(contexts)
    messages = [
    {
        "role": "system",
        "content": (
            "You are a concise Bengali assistant. "
            "Answer **only** using the provided context. "
            "If the context does not contain the answer, reply exactly: "
            "'উত্তর প্রসঙ্গে পাওয়া যায়নি।' "
            "Do not invent or guess."
        )
    },
    {
        "role": "user",
        "content": f"প্রসঙ্গ:\n{context_str}\n\nপ্রশ্ন: {req.question}"
    }
]
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            max_tokens=128,
            temperature=0.1
        )
        answer = chat_completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)