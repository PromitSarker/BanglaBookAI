# main.py
# FastAPI + Bengali PDF RAG (text → embeddings → Chroma → Groq)
# Uses bangla_pdf_ocr + heavy preprocessing v4.0
#   – added denoising, skew-correction, Tesseract page-segmentation
#   – vectors only in ChromaDB, text kept in ./chunk_store.json
#
# Requirements:
# pip install fastapi uvicorn python-multipart pymupdf pdfplumber opencv-python
#             pillow bangla-pdf-ocr sentence-transformers chromadb groq
#             python-dotenv pytesseract

import io
import json
import os
import pathlib
import re
import tempfile
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

# ML / Vector DB / LLM
import chromadb
import groq
import pytesseract
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uvicorn

# ---------- Environment ----------
load_dotenv()
app = FastAPI()

# ---------- Models & Clients ----------
model = SentenceTransformer("l3cube-pune/bengali-sentence-bert-nli")
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)
collection = chroma_client.get_or_create_collection(
    name="bengali_docs",
    metadata={"hnsw:space": "cosine"}
)
groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------- Tiny key-value store ----------
CHUNK_STORE = pathlib.Path("./chunk_store.json")
def _load_store() -> dict[str, str]:
    if CHUNK_STORE.exists():
        return json.loads(CHUNK_STORE.read_text(encoding="utf-8"))
    return {}
def _save_store(store: dict[str, str]) -> None:
    CHUNK_STORE.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")

# ---------- Text Cleaners ----------
_HIDDEN = re2.compile(r'[\u00AD\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF]+')
_BENGALI_ALLOWED = re2.compile(r'[^\p{Script=Bengali}\p{Bengali}\p{Nd}\p{P}\s]')
_LAT2BEN_DIGITS = str.maketrans('0123456789', '০১২৩৪৫৬৭৮৯')
_PUNCT_FIXES = str.maketrans({
    '\u2018': "'", '\u2019': "'",
    '\u201C': '"', '\u201D': '"',
    '\u2013': '-', '\u2014': '-', '\u2026': '...',
})
_DUP_PUNCT = re2.compile(r'([।!?,;:])\1{2,}')
_HYPHEN_BREAK = re2.compile(r'(\w+)-\s*\n\s*(\w+)')

def clean_bengali(text: str) -> str:
    text = ud.normalize("NFC", text)
    text = _HIDDEN.sub("", text)
    text = _BENGALI_ALLOWED.sub("", text)
    text = text.translate(_LAT2BEN_DIGITS)
    text = text.translate(_PUNCT_FIXES)
    text = re2.sub(r"\s+", " ", text)
    return text.strip()

def post_correct(text: str) -> str:
    # Insert your ligature_fix dict here if you have one
    _LIGATURE_FIXES = {
    # ----------------------------------------------------
    # I. Common Conjuncts (Basic Stacked/Horizontal/Modified)
    # ----------------------------------------------------
    "ক\u09CDক": "ক্ক",  # k + k
    "ক\u09CDষ": "ক্ষ",  # k + sh (very common, unique form)
    "গ\u09CDন": "গ্ন",  # g + n
    "ঙ\u09CDগ": "ঙ্গ",  # ng + g
    "ঙ\u09CDক": "ঙ্ক",  # ng + k
    "চ\u09CDচ": "চ্চ",  # ch + ch
    "চ\u09CDছ": "চ্ছ",  # ch + chh
    "জ\u09CDজ": "জ্জ",  # j + j
    "জ\u09CDঝ": "জ্ঝ",  # j + jh
    "জ\u09CDঞ": "জ্ঞ",  # j + n (often pronounced as g-y)
    "ঞ\u09CDজ": "ঞ্জ",  # n + j
    "ঞ\u09CDচ": "ঞ্চ",  # n + ch
    "ঞ\u09CDছ": "ঞ্ছ",  # n + chh
    "ট\u09CDট": "ট্ট",  # T + T
    "ড\u09CDড": "ড্ড",  # D + D
    "ণ\u09CDড": "ণ্ড",  # N + D
    "ণ\u09CDঠ": "ণ্ঠ",  # N + Th
    "ণ\u09CDঢ": "ণ্ঢ",  # N + Dh
    "ত\u09CDত": "ত্ত",  # t + t
    "ত\u09CDথ": "ত্থ",  # t + th
    "দ\u09CDদ": "দ্দ",  # d + d
    "দ\u09CDধ": "দ্ধ",  # d + dh
    "দ\u09CDব": "দ্ব",  # d + v/b
    "দ\u09CDভ": "দ্ভ",  # d + bh
    "ন\u09CDদ": "ন্দ",  # n + d
    "ন\u09CDধ": "ন্ধ",  # n + dh
    "ন\u09CDন": "ন্ন",  # n + n
    "প\u09CDপ": "প্প",  # p + p
    "ফ\u09CDফ": "ফ্ফ",  # ph + ph (sometimes treated as two separate)
    "ব\u09CDব": "ব্ব",  # b + b
    "ম\u09CDব": "ম্ব",  # m + b
    "ম\u09CDম": "ম্ম",  # m + m
    "য\u09CDয": "য্য",  # y + y
    "ল\u09CDল": "ল্ল",  # l + l
    "শ\u09CDচ": "শ্চ",  # sh + ch
    "শ\u09CDছ": "শ্ছ",  # sh + chh
    "স\u09CDস": "ಸ್ಸ",  # s + s
    "স\u09CDথ": "স্থ",  # s + th
    "হ\u09CDন": "হ্ন",  # h + n
    "হ\u09CDম": "হ্ম",  # h + m
    "হ\u09CDব": "হ্ব",  # h + b/v
    "হ\u09CDল": "হ্ল",  # h + l

    # ----------------------------------------------------
    # II. Conjuncts with R-phala (্র - U+09CD\u09B0 or just \u09B0 after virama)
    # The 'র' (ra) as a second consonant forms '্র' (r-phala)
    # ----------------------------------------------------
    "ক\u09CDর": "ক্র",  # k + r-phala
    "খ\u09CDর": "খ্র",  # kh + r-phala
    "গ\u09CDর": "গ্র",  # g + r-phala
    "ঘ\u09CDর": "ঘ্র",  # gh + r-phala
    "চ\u09CDর": "চ্র",  # ch + r-phala
    "ছ\u09CDর": "ছ্র",  # chh + r-phala
    "জ\u09CDর": "জ্র",  # j + r-phala
    "ট\u09CDর": "ট্র",  # T + r-phala (special form)
    "ঠ\u09CDর": "ঠ্র",  # Th + r-phala
    "ড\u09CDর": "ড্র",  # D + r-phala (special form)
    "ঢ\u09CDর": "ঢ্র",  # Dh + r-phala
    "ত\u09CDর": "ত্র",  # t + r-phala (unique form)
    "থ\u09CDর": "থ্র",  # th + r-phala
    "দ\u09CDর": "দ্র",  # d + r-phala
    "ধ\u09CDর": "ধ্র",  # dh + r-phala
    "ন\u09CDর": "ন্র",  # n + r-phala
    "প\u09CDর": "প্র",  # p + r-phala
    "ফ\u09CDর": "ফ্র",  # ph + r-phala
    "ব\u09CDর": "ব্র",  # b + r-phala
    "ভ\u09CDর": "ভ্র",  # bh + r-phala
    "ম\u09CDর": "ম্র",  # m + r-phala
    "শ\u09CDর": "শ্র",  # sh + r-phala
    "স\u09CDর": "স্র",  # s + r-phala
    "হ\u09CDর": "হ্র",  # h + r-phala

    # ----------------------------------------------------
    # III. Conjuncts with J-phala (্য - U+09CD\u09AF or just \u09AF after virama)
    # The 'য' (ya) as a second consonant forms '্য' (j-phala)
    # ----------------------------------------------------
    "ক\u09CDয": "ক্য",  # k + j-phala
    "গ\u09CDয": "গ্য",  # g + j-phala
    "ধ\u09CDয": "ধ্য",  # dh + j-phala
    "ন\u09CDয": "ন্য",  # n + j-phala
    "ম\u09CDয": "ম্য",  # m + j-phala
    "স\u09CDয": "স্য",  # s + j-phala
    "হ\u09CDয": "হ্য",  # h + j-phala

    # ----------------------------------------------------
    # IV. Conjuncts with M-phala (্ম - U+09CD\u09AE or just \u09AE after virama)
    # The 'ম' (ma) as a second consonant forms '্ম' (m-phala)
    # ----------------------------------------------------
    "ক\u09CDম": "কম",  # k + m-phala (often stacked vertically, represented as single glyph)
    "গ\u09CDম": "গম",  # g + m-phala
    "ড\u09CDম": "ডম",  # D + m-phala
    "ন\u09CDম": "নম",  # n + m-phala
    "স\u09CDম": "সম",  # s + m-phala

    # ----------------------------------------------------
    # V. Conjuncts with L-phala (্ল - U+09CD\u09B2 or just \u09B2 after virama)
    # The 'ল' (la) as a second consonant forms '্ল' (l-phala)
    # ----------------------------------------------------
    "ক\u09CDল": "কল",  # k + l-phala
    "প\u09CDল": "প্ল",  # p + l-phala
    "শ\u09CDল": "শ্ল",  # sh + l-phala

    # ----------------------------------------------------
    # VI. Reph (র্ - U+09B0\u09CD as first consonant)
    # The 'র' (ra) as a first consonant often forms 'র্' (reph) above the next consonant
    # ----------------------------------------------------
    "র\u09CDক": "র্ক",  # r + k
    "র\u09CDগ": "র্গ",  # r + g
    "র\u09CDচ": "র্চ",  # r + ch
    "র\u09CDজ": "র্জ",  # r + j
    "র\u09CDট": "র্ট",  # r + T
    "র\u09CDড": "র্ড",  # r + D
    "র\u09CDত": "র্ত",  # r + t
    "র\u09CDদ": "র্দ",  # r + d
    "র\u09CDধ": "র্ধ",  # r + dh
    "র\u09CDন": "র্ন",  # r + n
    "র\u09CDপ": "র্প",  # r + p
    "র\u09CDব": "র্ব",  # r + b
    "র\u09CDম": "র্ম",  # r + m
    "র\u09CDয": "র্য",  # r + y (reph + j-phala)
    "র\u09CDল": "র্ল",  # r + l
    "র\u09CDশ": "র্শ",  # r + sh
    "র\u09CDষ": "র্ষ",  # r + sh
    "র\u09CDস": "র্স",  # r + s
    "র\u09CDহ": "র্হ",  # r + h

    # ----------------------------------------------------
    # VII. Special/Complex Conjuncts (e.g., three consonants)
    # These often involve combinations of the above rules.
    # Not all three-consonant conjuncts have a single pre-composed Unicode character.
    # Often, they are rendered by the font engine combining a two-consonant conjunct with a third.
    # However, some might be explicitly defined for older systems or specific fonts.
    # Example for 'ন + দ + র':
    "ন\u09CDদ\u09CDর": "ন্দ্র", # n + d + r-phala (n + dr-phala)

    # ----------------------------------------------------
    # VIII. Some unique cases or less common conjuncts
    # ----------------------------------------------------
    "ড়\u09CDড়": "ড়্ড়", # This usually renders as two separate characters with virama
    "ঢ়\u09CDঢ়": "ঢ়্ঢ়", # This usually renders as two separate characters with virama
    "য়\u09CDয়": "য়্য", # A less common conjunct
    "য়\u09CDব": "য়্ব" # A less common conjunct
}
    text = _DUP_PUNCT.sub(r'\1', text)
    return text

def join_broken_words(text: str) -> str:
    return _HYPHEN_BREAK.sub(r'\1\2', text)

# ---------- NEW OCR v4 ----------
def _preprocess_v2(img_bgr: np.ndarray) -> np.ndarray:
    """Rich pre-processing: denoise, grayscale, deskew, binarize, cleanup."""
    # upscale
    img = cv2.resize(img_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # denoise
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    # deskew
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    if abs(angle) > 0.3:
        M = cv2.getRotationMatrix2D((gray.shape[1]//2, gray.shape[0]//2), angle, 1.0)
        gray = cv2.warpAffine(gray, M, gray.shape[::-1],
                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # CLAHE + Otsu binarization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary

def _tess_with_layout(bin_img: np.ndarray) -> str:
    """Tesseract with automatic page segmentation (PSM 3)."""
    data = pytesseract.image_to_data(
        Image.fromarray(bin_img),
        lang="ben",
        config="--psm 3",
        output_type=pytesseract.Output.DICT
    )
    lines, confs = [], []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        conf = int(data["conf"][i])
        if conf > 60 and word:
            lines.append(word)
            confs.append(conf)
    return " ".join(lines)

def run_bangla_ocr_on_png_bytes(img_bytes: bytes) -> str:
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    best_text, best_conf = "", 0.0

    # bangla_pdf_ocr first
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
        cv2.imwrite(tmp_in.name, _preprocess_v2(img_bgr))
        in_path = tmp_in.name
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp_out:
        out_path = tmp_out.name
    try:
        bpo_process_pdf(in_path, out_path)
        with open(out_path, encoding="utf-8") as f:
            raw = f.read()
    finally:
        os.unlink(in_path)
        os.unlink(out_path)

    cleaned = clean_bengali(raw)
    best_text, best_conf = cleaned, 85.0

    # Tesseract fallback with layout segmentation
    if best_conf < 90:
        bin_img = _preprocess_v2(img_bgr)
        tess_text = _tess_with_layout(bin_img)
        tess_cleaned = clean_bengali(tess_text)
        if tess_cleaned and len(tess_cleaned) > len(best_text) * 0.9:
            best_text = tess_cleaned

    joined = join_broken_words(best_text)
    corrected = post_correct(joined)
    return corrected

# ---------- Chunking ----------
def chunk_bengali(text: str, max_len: int = 600) -> List[str]:
    sentences = re2.split(r"(?<=[।!?])\s*", text)
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

# ---------- Render page to PNG ----------
def render_page_highres(page: fitz.Page) -> bytes:
    mat = fitz.Matrix(3.0, 3.0)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")

# ---------- FastAPI endpoints ----------
@app.post("/process")
async def process_pdf(file: UploadFile = File(...)):
    pdf_bytes = await file.read()

    # 1) pdfplumber first
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    text = clean_bengali(text)

    # 2) fallback
    if len(text) < 20:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_texts = []
        for page in doc:
            raw = page.get_text("text")
            if raw and len(raw.strip()) > 20:
                page_texts.append(clean_bengali(raw))
            else:
                img_bytes = render_page_highres(page)
                page_texts.append(run_bangla_ocr_on_png_bytes(img_bytes))
        doc.close()
        text = " ".join(page_texts)

    chunks = chunk_bengali(text)
    embeddings = model.encode(chunks, normalize_embeddings=True).tolist()
    ids = [f"{file.filename}_{i}" for i in range(len(chunks))]

    # store vectors only in ChromaDB
    collection.add(embeddings=embeddings, ids=ids)

    # store text locally
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
    prompt = (
        f"প্রসঙ্গ: {context_str}\n\n"
        f"প্রশ্ন: {req.question}\n\n"
        "উত্তর শুধু এক লাইনে বাংলায় দাও, ব্যাখ্যা বা বাড়তি কিছু নয়:"
    )

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.1,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"answer": answer}

# ---------- Entry ----------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)