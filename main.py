# main.py
# FastAPI + Bengali PDF RAG (text → embeddings → Chroma → Groq)
# Uses bangla_pdf_ocr + heavy preprocessing v3.0 (EDSR super-res + Tesseract fallback)

# ---------- Requirements ----------
# pip install fastapi uvicorn python-multipart \
#             pymupdf pdfplumber opencv-python pillow \
#             bangla-pdf-ocr sentence-transformers chromadb groq \
#             regex python-dotenv pytesseract

# ---------- Imports ----------
import os
import io
import re
import tempfile
from typing import List

import cv2
import fitz  # PyMuPDF
import numpy as np
import pdfplumber
import unicodedata as ud
import regex as re2
from bangla_pdf_ocr import process_pdf as bpo_process_pdf
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageEnhance
from pydantic import BaseModel

# ML / Vector DB / LLM
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import groq
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

# ---------- Text Cleaners 2.0 ----------
_HIDDEN = re2.compile(r'[\u00AD\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF]+')
_BENGALI_ALLOWED = re2.compile(r'[^\p{Script=Bengali}\p{Bengali}\p{Nd}\p{P}\s]')
_LAT2BEN_DIGITS = str.maketrans('0123456789', '০১২৩৪৫৬৭৮৯')

_LIGATURE_FIXES = {
    # --- যুক্তবর্ন / ligature fixes ---
    'ত্ত': 'ত্ত', 'ত্ত্য': 'ত্ত্য', 'ন্ত্য': 'ন্ত্য',
    'ঙ্ক': 'ঙ্ক', 'ঙ্গ': 'ঙ্গ', 'ঞ্চ': 'ঞ্চ', 'ঞ্জ': 'ঞ্জ',
    'ণ্ড': 'ণ্ড', 'ণ্ট': 'ণ্ট', 'ষ্ট': 'ষ্ট', 'ষ্ঠ': 'ষ্ঠ',
    'স্ত': 'স্ত', 'স্থ': 'স্থ', 'স্প': 'স্প', 'স্ক': 'স্ক',
    'দ্গ': 'দ্গ', 'দ্দ': 'দ্দ', 'দ্ধ': 'দ্ধ', 'ন্দ': 'ন্দ',
    'র্থ': 'র্থ', 'র্ত': 'র্ত', 'র্প': 'র্প', 'র্ম': 'র্ম',
    'র্শ': 'র্শ', 'র্ষ': 'র্ষ', 'র্ন': 'র্ন', 'র্ল': 'র্ল',
    'র্ব': 'র্ব', 'র্হ': 'র্হ', 'র্ফ': 'র্ফ', 'র্ড': 'র্ড',
    'র্ট': 'র্ট', 'র্দ': 'র্দ', 'র্গ': 'র্গ', 'র্ঘ': 'র্ঘ',
    'র্চ': 'র্চ', 'র্জ': 'র্জ', 'র্ণ': 'র্ণ', 'র্ন্দ': 'র্ন্দ',
    'র্ন্ত': 'র্ন্ত', 'র্ব্য': 'র্ব্য', 'র্স্য': 'র্স্য',
    'ক্ষ': 'ক্ষ', 'ক্ষ্ম': 'ক্ষ্ম', 'জ্ঞ': 'জ্ঞ',
    'শ্চ': 'শ্চ', 'শ্ন': 'শ্ন', 'শ্ম': 'শ্ম', 'শ্ল': 'শ্ল',
    'স্ত্ব': 'স্ত্ব', 'স্ত্ম': 'স্ত্ম', 'স্ন': 'স্ন', 'স্ম': 'স্ম',
    'স্ল': 'স্ল', 'স্খ': 'স্খ', 'স্ফ': 'স্ফ', 'স্ব': 'স্ব',
    'স্ম্য': 'স্ম্য', 'স্ল্য': 'স্ল্য', 'স্ব্য': 'স্ব্য',

    # --- ভুল কার-জোড়া / vowel-mark order fixes ---
    'াে': 'ো',  'ো': 'ো',
    'েি': 'ৈ',  'িে': 'ৈ',
    'াৈ': 'ো',  'ৈা': 'ো',  # occasional PDF split
    'ুে': 'ু',   'েু': 'ু',
    'ূে': 'ূ',   'েূ': 'ূ',
    'ৃে': 'ৃ',   'েৃ': 'ৃ',
    'েে': 'ে',   # doubled e-kar
    'াা': 'া',   # doubled a-kar
    'িি': 'ি',   # doubled i-kar
    'ুু': 'ু',   # doubled u-kar
    'ূূ': 'ূ',   # doubled uu-kar
    'ৃৃ': 'ৃ',   # doubled rri-kar
    'েো': 'ো',   # e + o → o-kar
    'োে': 'ো',   # o + e → o-kar
    'ৈে': 'ৈ',   # oi + e → oi-kar
    'েৈ': 'ৈ',

    # --- হসন্ত / hasant fixes ---
    '্‌': '্',   # Zero-Width Non-Joiner after hasant
    '‌্': '্',   # ZWNJ before hasant
    '্‍': '্',   # Zero-Width Joiner after hasant
    '‍্': '্',   # ZWJ before hasant
    '্্': '্',   # double hasant
}
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
    for wrong, right in _LIGATURE_FIXES.items():
        text = text.replace(wrong, right)
    text = _DUP_PUNCT.sub(r'\1', text)
    return text

def join_broken_words(text: str) -> str:
    text = _HYPHEN_BREAK.sub(r'\1\2', text)
    return text

# ---------- OCR helpers v3 ----------
import pytesseract   # pip install pytesseract
from PIL import ImageEnhance

# 1. Super-resolution helper (waifu2x-CV2)
def _sr(img: np.ndarray) -> np.ndarray:
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("EDSR_x4.pb")           # https://github.com/Saafke/EDSR_Tensorflow
    sr.setModel("edsr", 4)
    return sr.upsample(img)

# 2. Heavy pre-processing chain
def _preprocess(img: np.ndarray, variant: int = 0) -> np.ndarray:
    # a) Upscale 2×
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # b) CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    # c) Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    if variant == 0:
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)   # remove pepper
    else:
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # fill gaps

    # d) Otsu
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # e) Deskew
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) > 0.5:
        M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, img.shape[::-1],
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    return img

# 3. Confidence filter for Tesseract
def _tess_clean(data: dict) -> str:
    lines, confs = [], []
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        conf = int(data["conf"][i])
        if conf > 70 and word:
            lines.append(word)
            confs.append(conf)
    return " ".join(lines), (sum(confs) / len(confs)) if confs else 0.0

# 4. Main OCR function
def run_bangla_ocr_on_png_bytes(img_bytes: bytes) -> str:
    # PIL → OpenCV
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    best_text, best_conf = "", 0.0

    # —— Try bangla_pdf_ocr first (original pipeline) ——
    buf = io.BytesIO()
    Image.fromarray(_preprocess(img, 0)).save(buf, format="PNG")
    buf.seek(0)
    pre_bytes = buf.read()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
        tmp_in.write(pre_bytes)
        in_path = tmp_in.name
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp_out:
        out_path = tmp_out.name

    try:
        bpo_process_pdf(in_path, out_path)
        with open(out_path, encoding="utf-8") as f:
            raw1 = f.read()
    finally:
        os.unlink(in_path)
        os.unlink(out_path)

    raw1 = clean_bengali(raw1)
    best_text, best_conf = raw1, 85.0  # bangla_pdf_ocr gives no conf → assume OK

    # —— Fallback to Tesseract if confidence < 90 ——
    if best_conf < 90:
        for variant in (0, 1):
            processed = _preprocess(img, variant)
            tess_txt = pytesseract.image_to_data(
                Image.fromarray(processed),
                lang="ben",
                output_type=pytesseract.Output.DICT
            )
            txt, conf = _tess_clean(tess_txt)
            if conf > best_conf:
                best_text, best_conf = txt, conf
            if best_conf >= 90:
                break

    cleaned = clean_bengali(best_text)
    joined = join_broken_words(cleaned)
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

    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    return {"chunks": chunks, "stored_count": len(chunks)}

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
        f"প্রসঙ্গ:\n{context_str}\n\n"
        f"প্রশ্ন: {req.question}\nউত্তর:"
    )

    try:
        resp = groq_client.chat.completions.create(
            model="qwen-3-32b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.15,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"answer": answer}

# ---------- Entry ----------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)