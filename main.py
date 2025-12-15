# main.py — LumoChart Backend (v3.1 optimized)
# Fast, accurate, and production-ready

import os, io, re, uuid, json, asyncio
from datetime import datetime
from typing import Optional, Dict, Any, Literal, List
from enum import Enum
from urllib.parse import quote

from fastapi import (
    FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request, APIRouter
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from pydantic import BaseModel, Field

from dotenv import load_dotenv
import firebase_admin
from firebase_admin import storage, firestore

import httpx
from openai import AsyncOpenAI, RateLimitError, APIError

# ✅ Routers that are known-safe
from subscriptions_main import router as subscriptions_router


# ─────────────────────────────────────────────────────────────
# SETUP & INITIALIZATION
# ─────────────────────────────────────────────────────────────

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "lumochart-backend")
BUCKET_NAME = os.getenv("FIREBASE_BUCKET", "lumochart.firebasestorage.app")

try:
    firebase_admin.initialize_app(
        options={
            "storageBucket": BUCKET_NAME,
            "projectId": PROJECT_ID,
        }
    )
except ValueError:
    print("⚠️ Firebase already initialized.")

db = firestore.Client(project=PROJECT_ID)
bucket = storage.bucket()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ CREATE APP FIRST (critical)
app = FastAPI()


# ─────────────────────────────────────────────
# API VERSIONING
# ─────────────────────────────────────────────

v1 = APIRouter(prefix="/v1")
v2 = APIRouter(prefix="/v2")


# ─────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://chart.lumosyehealth.com",
    "https://www.lumosyehealth.com",
    "https://lumosyehealth.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# ROUTERS
# ─────────────────────────────────────────────

# Always-safe router
app.include_router(subscriptions_router)


# ─────────────────────────────────────────────
# UNIVERSAL PREFLIGHT (Cloud Run safe)
# ─────────────────────────────────────────────

@app.options("/{rest_of_path:path}")
async def universal_preflight(request: Request, rest_of_path: str):
    origin = request.headers.get("origin", "*")
    response = Response()
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

# ─────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────

class NoteFields(BaseModel):
    hpi: Optional[str] = Field(None)
    review_of_systems: Optional[str] = Field(None)
    past_medical_history: Optional[str] = None
    past_surgical_history: Optional[str] = None
    social_history: Optional[str] = None
    family_history: Optional[str] = None
    medications: Optional[str] = None
    allergies: Optional[str] = None
    exam: Optional[str] = None
    objective: Optional[str] = None
    assessment_and_plan: Optional[str] = Field(None, alias="ap")
    mdm: Optional[str] = None
    disposition: Optional[str] = None
    consult_question: Optional[str] = None
    interval_history: Optional[str] = None
    class Config:
        populate_by_name = True

class EDHistory(BaseModel):
    hpi: str
    review_of_systems: Optional[str] = None

class EDInitialCaptureResponse(BaseModel):
    transcript: str
    history: EDHistory

class CaptureInputType(str, Enum):
    audio_transcript = "audio_transcript"
    image_base64 = "image_base64"
    text_snippet = "text_snippet"

class CaptureInput(BaseModel):
    type: CaptureInputType
    content: str

class SmartCaptureRequest(BaseModel):
    note_type: str
    inputs: List[CaptureInput]

class JobStartRequest(BaseModel):
    file_path: str

class JobStatusResponse(BaseModel):
    status: str
    message: Optional[str] = None
    transcript: Optional[str] = None

class TranscriptionResponse(BaseModel):
    text: str


class GenerateAPV2Request(BaseModel):
    context: str
    current_note_context: str
    past_notes_context: Optional[str] = ""
    consult_question: Optional[str] = None
    patient_factors: Optional[str] = None

class MergeNoteRequest(BaseModel):
    note_type: str
    existing_note: dict
    new_text: str

class DrugSearchResponse(BaseModel):
    results: List[Dict[str, Any]]

class Note(BaseModel):
    timestamp: datetime
    type: str
    hpi: str | None = None
    subjective: str | None = None
    exam: str | None = None
    objective: str | None = None
    ap: str | None = None
    mdm: str | None = None   # ✅ add this line

class SummaryResponse(BaseModel):
    summary: str

class GenerateMDMRequest(BaseModel):
    hpi: Optional[str] = None
    exam: Optional[str] = None
    objective: Optional[str] = None # For labs/imaging
    assessment: Optional[str] = None # The "A" part of A&P, without the plan

class NotesPayload(BaseModel):
    notes: List[Note]

class FormatHPIRequest(BaseModel):
    hpi: Optional[str] = None
    pmh: Optional[str] = None
    ros: Optional[str] = None
    patient_name: Optional[str] = None
    sex: Optional[str] = None   # "male", "female", or None
    age: Optional[int] = None

class FormatHPIResponse(BaseModel):
    formatted_hpi: str

class GenerateHPIRequest(BaseModel):
    # Match your NoteContext rawValue options
    note_type: Literal[
        "admission",
        "progress",
        "consult",
        "outpatient",
        "ed",
        "bedside"
    ]

    # Raw context text from the note (HPI/ROS/PMH/etc.)
    context: str

    # NEW — Optional demographic enhancers
    patient_name: Optional[str] = None
    sex: Optional[str] = None  # "male", "female", etc.
    age: Optional[int] = None


class GenerateHPIResponse(BaseModel):
    hpi: str

    
# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

def log_event(level: str, event: str, **details):
    print(json.dumps({
        "severity": level,
        "event": event,
        "timestamp": datetime.utcnow().isoformat(),
        **details
    }))

@app.get("/")
async def root():
    return {"message": "LumoChart Backend v3.2 running ✅"}

@app.get("/firestore-test")
async def firestore_test():
    test_ref = db.collection("test_connection").document("ping")
    test_ref.set({"timestamp": datetime.utcnow().isoformat(), "message": "connected ✅"})
    doc = test_ref.get()
    return {"status": "ok", "data": doc.to_dict()}


# ---------- Helper: map note_type -> style hint ----------

def _hpi_style_for_note_type(note_type: str) -> str:
    mapping = {
        "admission": "inpatient hospital admission",
        "progress": "inpatient daily progress note",
        "consult": "inpatient consultation",
        "outpatient": "outpatient clinic visit",
        "ed": "emergency department encounter",
        "bedside": "brief bedside update",
    }
    return mapping.get(note_type, "general medical encounter")


# ---------- Helper: scrub headings / extra labels ----------

_HPI_HEADING_PATTERN = re.compile(
    r"^\s*(hpi|history of present illness)\s*[:\-–]\s*",
    flags=re.IGNORECASE
)

def _clean_hpi_text(text: str) -> str:
    t = text.strip()

    # Drop "HPI:", "History of Present Illness:" heading if present
    t = _HPI_HEADING_PATTERN.sub("", t).strip()

    # Optionally strip obvious section labels if the model misbehaves
    # e.g. "Assessment:", "Plan:", "ED Course:", etc.
    # We *only* remove them if they appear as a standalone heading at the top.
    for heading in ["Assessment", "Plan", "ED Course", "MDM", "Exam", "ROS"]:
        pattern = re.compile(
            rf"^\s*{heading}\s*[:\-–]\s*",
            flags=re.IGNORECASE
        )
        t = pattern.sub("", t).strip()

    return t


# ─────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────

async def safe_ai_call(**kwargs):
    """Unified retry wrapper for OpenAI calls."""
    for attempt in range(3):
        try:
            return await client.chat.completions.create(**kwargs)
        except (RateLimitError, APIError) as e:
            wait = 2 ** attempt
            log_event("WARNING", "ai_retry", attempt=attempt+1, wait=wait, error=str(e))
            await asyncio.sleep(wait)
    raise HTTPException(502, detail="AI service unavailable after retries")

async def ai_completion_or_fail(**kwargs):
    """
    Wrapper identical to safe_ai_call but throws explicit HTTPException(500)
    if the OpenAI call fails. Keeps backward compatibility with older endpoints.
    """
    try:
        return await client.chat.completions.create(**kwargs)
    except Exception as e:
        log_event("ERROR", "ai_completion_failed", error=str(e))
        raise HTTPException(500, f"AI completion failed: {e}")
        
# ─────────────────────────────────────────────────────────────
# TRANSCRIPTION (Now using Firestore)
# ─────────────────────────────────────────────────────────────

@v1.post("/start-transcription-job")
async def start_transcription_job(request: JobStartRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    job_ref = db.collection("transcription_jobs").document(job_id)
    
    # Create the job document in Firestore
    job_ref.set({
        "status": "queued",
        "message": "Job received",
        "transcript": None,
        "createdAt": firestore.SERVER_TIMESTAMP
    })
    
    background_tasks.add_task(process_transcription_in_background, job_id, request.file_path)
    return {"status": "processing", "job_id": job_id}


async def process_transcription_in_background(job_id: str, file_path: str):
    job_ref = db.collection("transcription_jobs").document(job_id)

    def update_job(status, msg, transcript=None):
        job_ref.update({
            "status": status,
            "message": msg,
            "transcript": transcript
        })

    try:
        update_job("processing", "Downloading from Firebase...")
        audio_bytes = bucket.blob(file_path).download_as_bytes()
        
        update_job("processing", "Transcribing with Whisper...")
        audio_buf = io.BytesIO(audio_bytes)
        audio_buf.name = "audio.m4a"
        
        result = await client.audio.transcriptions.create(model="whisper-1", file=audio_buf)
        update_job("complete", "Success", result.text.strip())
        
    except Exception as e:
        error_message = f"Transcription error: {str(e)}"
        update_job("failed", error_message)
        log_event("ERROR", "transcription_failed", job_id=job_id, error=str(e))


@v1.get("/get-job-status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    job_ref = db.collection("transcription_jobs").document(job_id)
    doc = job_ref.get()
    
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return JobStatusResponse(**doc.to_dict())

@v1.post("/transcribe-objective", response_model=TranscriptionResponse)
async def transcribe_objective(file: UploadFile = File(...)):
    contents = await file.read()
    buf = io.BytesIO(contents); buf.name = file.filename
    try:
        res = await client.audio.transcriptions.create(model="whisper-1", file=buf)
        return TranscriptionResponse(text=res.text)
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")

# ─────────────────────────────────────────────────────────────
# AI NOTE GENERATION (Final – Demographic-Aware & Natural Narrative)
# ─────────────────────────────────────────────────────────────

MASTER_SYSTEM_PROMPT = """
You are an expert clinician and documentation specialist. Transform raw transcripts and clinical context into a structured medical note.

Rules:
1. Do NOT invent facts. If unknown, return null for that field.
2. Use concise, formal, clinical language.
3. Return ONLY a valid JSON object (no markdown, comments, or explanations).
4. Each problem’s plan must include actionable details (drug, dose, route, frequency, duration).
5. Avoid repeating the same information in multiple sections (e.g., don’t restate PMH items already covered in HPI).
6. If demographics are provided, begin the HPI naturally:
   “Mr./Ms. [Name] is a [Age]-year-old [sex] patient with a past medical history of … who presents with …”
7. If only partial demographics are available (e.g., missing name or sex), use the most natural fallback form, e.g.:
   “The patient is a [Age]-year-old male who presents with …” or “The patient presents with …”
"""

@v1.post("/generate-structured-note")
async def generate_structured_note(request: SmartCaptureRequest):
    log_event("INFO", "note_generation_start", note_type=request.note_type)
    consolidated, image_tasks = [], []

    # Gather multimodal inputs
    for item in request.inputs:
        if item.type in [CaptureInputType.audio_transcript, CaptureInputType.text_snippet]:
            consolidated.append(item.content)
        elif item.type == CaptureInputType.image_base64:
            img_prompt = "Analyze this image in clinical context. Extract only clinically relevant findings (e.g., vitals, labs, imaging impressions)."
            image_tasks.append(
                safe_ai_call(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": img_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{item.content}"}}
                    ]}],
                    max_completion_tokens=1024
                )
            )

    # Run Vision tasks concurrently
    if image_tasks:
        results = await asyncio.gather(*image_tasks, return_exceptions=True)
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                consolidated.append(f"[Image {i+1} failed: {str(r)}]")
            else:
                consolidated.append(f"--- Image {i+1} ---\n{r.choices[0].message.content.strip()}")

    full_context = "\n\n".join(consolidated).strip()
    if not full_context:
        raise HTTPException(400, "No input content provided")

    # 🧩 Demographic context builder with fallbacks
    demographics_text = "Demographics unavailable."
    if hasattr(request, "patient_info") and request.patient_info:
        p = request.patient_info
        name = p.name or ""
        age = f"{p.age}-year-old " if p.age else ""
        sex = p.sex.lower() if p.sex else ""

        if name and age and sex:
            demographics_text = f"{name}, a {age}{sex} patient"
        elif age and sex:
            demographics_text = f"A {age}{sex} patient"
        elif age:
            demographics_text = f"A {age}patient"
        elif sex:
            demographics_text = f"A {sex} patient"
        else:
            demographics_text = "The patient"

    # Reject too short inputs
    if len(full_context.split()) < 25:
        raise HTTPException(400, "Input too short for structured note generation")

    try:
        # 🔹 Special ED path
        if request.note_type.lower() == "ed":
            prompt = f"""
You are an emergency medicine physician. Using the transcript and provided demographics, extract a concise HPI and structured Review of Systems.

Start the HPI naturally, e.g.:
“Mr./Ms. [Name] is a [Age]-year-old [sex] patient with a past medical history of … who presents with …”
If demographics are incomplete, use natural fallback phrasing (e.g., “The patient presents with …”).

Return valid JSON:
{{
  "hpi": "...",
  "review_of_systems": "..."
}}

Demographics context: {demographics_text}
"""
            resp = await safe_ai_call(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": full_context},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_completion_tokens=800,
            )
            data = json.loads(resp.choices[0].message.content)
            return EDInitialCaptureResponse(transcript=full_context, history=EDHistory(**data))

        # 🔹 Default path (inpatient, consult, outpatient, etc.)
        schema_text = json.dumps(NoteFields.model_json_schema()["properties"], indent=2)
        user_prompt = f"""
Demographics (if available): {demographics_text}

Generate a structured {request.note_type} note from this context.
Follow the schema below and ensure all fields conform to JSON structure:
{schema_text}

Context:
---
{full_context}
---
"""

        resp = await safe_ai_call(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": MASTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_completion_tokens=1500,
        )

        json_data = json.loads(resp.choices[0].message.content)
        return NoteFields(**json_data)

    except Exception as e:
        log_event("ERROR", "note_generation_failed", error=str(e))
        raise HTTPException(500, f"Note generation failed: {e}")


# ---------- HPI Generator Endpoint (Demographics-Aware, Improved) ----------

@v1.post("/generate-hpi", response_model=GenerateHPIResponse)
async def generate_hpi(req: GenerateHPIRequest) -> GenerateHPIResponse:
    """
    Generate a narrative HPI from structured/context text.

    NEW FEATURES:
    • Uses demographics (name, age, sex) to build a natural first sentence.
    • Automatically chooses Mr./Ms. if applicable.
    • Falls back gracefully if some demographics are missing.
    • Preserves all safety rules: NO plan, NO exam, NO disposition, NO hallucinated facts.
    """

    # --- Validate context ---
    context = (req.context or "").strip()
    if not context:
        raise HTTPException(status_code=400, detail="Context is empty.")

    style_hint = _hpi_style_for_note_type(req.note_type)

    # --- Build demographic identity line ---
    name = req.patient_name
    age = req.age
    sex = req.sex.lower() if req.sex else None

    # Honorific (optional, inferred)
    honorific = None
    if sex:
        if sex.startswith("m"): honorific = "Mr."
        elif sex.startswith("f"): honorific = "Ms."

    # Construct identity fragment
    if name and age and sex:
        identity = f"{honorific} {name} is a {age}-year-old {sex} patient"
    elif age and sex:
        identity = f"A {age}-year-old {sex} patient"
    elif age:
        identity = f"A {age}-year-old patient"
    elif sex:
        identity = f"A {sex} patient"
    else:
        identity = "The patient"

    # --------------------------
    # SYSTEM PROMPT (Upgraded)
    # --------------------------
    system_msg = f"""
You are a medical documentation assistant who writes high-quality
History of Present Illness (HPI) narratives for clinicians.

SAFETY & RULES:
• Use ONLY the clinical details explicitly present in the provided context.
• DO NOT invent diagnoses, labs, imaging, PMH, treatments, or timelines.
• DO NOT include exam findings, assessment, plan, differential, or disposition.
• DO NOT mention your process, AI, reasoning, or limitations.
• Do NOT include headings like "HPI:" or "History of Present Illness:".

HPI STRUCTURE:
• 1–3 concise narrative paragraphs.
• Third-person clinical tone.
• The FIRST SENTENCE MUST begin with:
    “{identity} … who presents with …”
  and should incorporate any PMH **only if explicitly present in context**.
• Subsequent sentences should cover:
    – onset
    – duration
    – location
    – quality
    – severity
    – associated symptoms
    – relevant negatives
    – contextual events
• If some details are missing, simply omit them (do not guess).

The output must be ONLY the polished HPI text.
"""

    # --------------------------
    # USER PROMPT
    # --------------------------
    user_msg = f"""
NOTE TYPE: {style_hint}

Using ONLY the information below, write the HPI as instructed.

CONTEXT START
{context}
CONTEXT END
"""

    # --------------------------
    # OpenAI Call
    # --------------------------
    try:
        completion = await client.chat.completions.create(
            model=os.getenv("OPENAI_HPI_MODEL", "gpt-4o-mini"),
            temperature=0.25,
            max_tokens=600,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI HPI generation failed: {e}"
        )

    raw = (completion.choices[0].message.content or "").strip()
    cleaned = _clean_hpi_text(raw)

    if not cleaned:
        raise HTTPException(
            status_code=500,
            detail="Generated HPI was empty after cleaning."
        )

    return GenerateHPIResponse(hpi=cleaned)


# ─────────────────────────────────────────────────────────────
# Format HPI
# ─────────────────────────────────────────────────────────────
@v1.post("/format-hpi", response_model=FormatHPIResponse)
async def format_hpi(req: FormatHPIRequest):
    """
    Rewrites a raw or disorganized HPI into a professional narrative.
    Incorporates known demographics (name, age, sex) from app context if available.
    """

    # 🔹 Inline synonym map (fast reference)
    clinical_synonyms = """
    Common Lay → Medical Equivalent:
    - blood clot in lungs → pulmonary embolism
    - clot in leg → deep vein thrombosis (DVT)
    - blood thinner → anticoagulant
    - heart attack → myocardial infarction
    - high blood pressure → hypertension
    - low blood pressure → hypotension
    - sugar → diabetes mellitus
    - thyroid problem → thyroid disorder
    - stroke → cerebrovascular accident
    - cancer → malignancy
    - kidney failure → renal failure
    - liver disease → hepatic disease
    - asthma attack → asthma exacerbation
    - lung infection → pneumonia
    - urinary infection → urinary tract infection (UTI)
    - black stool → melena
    - blood in stool → hematochezia
    - blood in urine → hematuria
    """

    # ✅ Determine honorific automatically if not provided in text
    honorific = None
    if req.sex:
        honorific = "Mr." if req.sex.lower().startswith("m") else "Ms."
    if req.patient_name and req.patient_name.lower().startswith(("miss", "ms", "mrs", "mr")):
        honorific = None  # Already included

    # ✅ Build contextual identity line for prompt
    demographics_summary = ""
    if req.patient_name or req.sex or req.age:
        name_part = req.patient_name or "the patient"
        age_part = f"{req.age}-year-old " if req.age else ""
        sex_part = f"{req.sex.lower()} " if req.sex else ""
        if honorific:
            demographics_summary = f"{honorific} {name_part} is a {age_part}{sex_part.strip()} patient"
        else:
            demographics_summary = f"{name_part} is a {age_part}{sex_part.strip()} patient"
    else:
        demographics_summary = "The patient"

    # ✅ Improved natural phrasing guidance
    system_prompt = f"""
    You are an experienced hospitalist and documentation specialist.
    Your goal is to rewrite the HPI into a concise, chronological, and professional narrative.

    🎯 OUTPUT REQUIREMENTS:
    - Begin with a natural, standardized sentence describing the patient (use the demographics if available).
      Example: "{demographics_summary} with a past medical history of hypertension and diabetes who presents with chest pain and shortness of breath."
    - Integrate relevant PMH and ROS smoothly into the opening sentence.
    - Replace any lay terms using this synonym map:
    {clinical_synonyms}
    - Keep the paragraph 4–6 sentences long, formal, and factual.
    - Focus only on subjective information.
    - Output only the rewritten paragraph.
    """

    user_prompt = f"""
    Provided data:
    HPI text: {req.hpi or 'Not provided.'}
    Past Medical History: {req.pmh or 'Not provided.'}
    Review of Systems: {req.ros or 'Not provided.'}
    Demographics: {demographics_summary}
    """

    resp = await safe_ai_call(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=700
    )

    formatted = resp.choices[0].message.content.strip()
    return FormatHPIResponse(formatted_hpi=formatted)


# ─── ED SUMMARY GENERATOR (polished & bullet formatted) ─────────────────────────

class EDSummaryResponse(BaseModel):
    assessment_and_plan: str
    mdm: str
    disposition_recommendation: Optional[str] = None
    billing_level_suggestion: Optional[str] = None


@v1.post("/generate-ed-summary", response_model=EDSummaryResponse)
async def generate_ed_summary(request: TranscriptionResponse):
    """
    Generates a polished ED Assessment & Plan, MDM, Disposition, and Billing Suggestion
    from a summarized transcript (HPI + ROS + Exam).
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing transcript text")

    system_prompt = """
You are an experienced emergency medicine physician documenting an ED encounter.

TASK:
From the provided HPI, ROS, and Exam text, generate the following structured JSON fields:

{
  "assessment_and_plan": "...",
  "mdm": "...",
  "disposition_recommendation": "...",
  "billing_level_suggestion": "..."
}

ASSESSMENT & PLAN — REQUIRED FORMAT (strict):
- Must be problem-based.
- Each problem must start on its own line with:
      #1 Diagnosis
      #2 Next Diagnosis
      #3 ...
- Each problem must contain one or more plan bullet points, each on its own line:
      - Plan item 1
      - Plan item 2
      - Plan item 3
- ABSOLUTELY NO middle dots (•). Use hyphen-minus bullets only.
- NO inline A&P. NO single-paragraph A&P. NO collapsing problems.
- The structure **must look exactly like this**:

#1 Breakthrough seizure
- Continue AED regimen
- Neurology consult
- Educate on seizure precautions

#2 Migraine headache
- IV fluids
- Antiemetic
- NSAID or triptan for acute relief

MDM — RULES:
- One formal paragraph.
- Describe differential diagnosis, key data reviewed, and reasoning.
- First sentence MUST begin with:
  “The patient is a [age]-year-old [sex] with a history of … who presents with …”

DISPOSITION — RULES:
- One concise line: Admit / Discharge / Observation + brief justification.

BILLING — RULES:
- Must return exactly one CPT: 99283, 99284, or 99285.

GENERAL RULES:
- Use concise, formal, clinical language.
- Do NOT hallucinate diagnoses or findings.
- Return ONLY valid JSON with the keys specified.
- No markdown, headings, or explanations outside the JSON.
"""

    # Call GPT-4o safely
    resp = await safe_ai_call(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0.35,
        max_completion_tokens=1300,
    )

    data = json.loads(resp.choices[0].message.content)

    # Ensure all keys exist for decoding
    data.setdefault("assessment_and_plan", "")
    data.setdefault("mdm", "")
    data.setdefault("disposition_recommendation", data.get("disposition", ""))
    data.setdefault("billing_level_suggestion", data.get("billing_suggestion", ""))

    # Align output keys with Swift model
    mapped = {
        "assessment_and_plan": data.get("assessment_and_plan", ""),
        "mdm": data.get("mdm", ""),
        "disposition_recommendation": data.get("disposition_recommendation") or data.get("disposition") or "",
        "billing_level_suggestion": data.get("billing_level_suggestion") or data.get("billing_suggestion") or ""
    }

    return EDSummaryResponse(**mapped)

# ─────────────────────────────────────────────────────────────
# A&P GENERATOR (v2)
# ─────────────────────────────────────────────────────────────

@v1.post("/generate-ap-v2", response_model=SummaryResponse)
async def generate_ap_v2(request: GenerateAPV2Request):
    sys_prompt = f"""
You are an attending physician crafting a problem-based Assessment & Plan.
Context: {request.context.upper()}
Rules:
- Start with 'Assessment:' then number each problem (#1., #2.).
- Each problem: short assessment + bulleted plan items (•).
- Include drug, dose, route, frequency, duration, and rationale.
"""
    user = f"CURRENT NOTE:\n{request.current_note_context}\n\nPAST NOTES:\n{request.past_notes_context or 'N/A'}"
    resp = await safe_ai_call(
        model="gpt-4o",
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
        temperature=0.1,
        max_completion_tokens=800
    )
    return SummaryResponse(summary=resp.choices[0].message.content.strip())

# ─────────────────────────────────────────────────────────────
# NOTE MERGE
# ─────────────────────────────────────────────────────────────

@v1.post("/merge-note-from-text", response_model=NoteFields)
async def merge_note_from_text(request: MergeNoteRequest):
    schema_text = json.dumps(NoteFields.model_json_schema()["properties"], indent=2)
    prompt = f"Merge NEW TEXT into EXISTING NOTE for a {request.note_type} note. Keep original facts unless clearly replaced. Output JSON matching this schema:\n{schema_text}\n\nEXISTING NOTE:\n{json.dumps(request.existing_note, indent=2)}\n---\nNEW TEXT:\n{request.new_text}"
    resp = await safe_ai_call(
        model="gpt-4o",
        messages=[{"role": "system", "content": MASTER_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_completion_tokens=1200
    )
    data = json.loads(resp.choices[0].message.content)
    return NoteFields(**data)

# ─── PHYSICAL EXAM FORMATTER ─────────────────────────────────────────────────────

@v1.post("/format-exam", response_model=SummaryResponse)
async def format_exam(request: TranscriptionResponse):
    """
    Formats a free-text physical exam into a clean, concise, sectioned format,
    using single-line sections (no bullet points).
    """
    system_prompt = """
You are a clinical documentation assistant. Format the following physical exam
into a clean, sectioned narrative style for a physician note.

✅ Sections to use (include only those relevant):
Vitals
General
HEENT
Respiratory
Cardiovascular
Gastrointestinal
Genitourinary
Musculoskeletal
Neurological
Skin

✅ Formatting Rules:
- Use one line per section.
- Do NOT use bullets, hyphens, or list markers.
- Each line should start with the section name followed by a colon, then concise comma-separated findings.
- Combine multiple vitals or findings on the same line.
- Keep phrasing short and clinical (e.g., “RRR, normal S1S2”).
- Preserve all findings exactly as provided — do not infer or remove information.
- Return plain text (no Markdown, no JSON, no titles).

Example Format:
Vitals: BP 130/90, HR 100, T 98F, BMI 30.4
General: No acute distress
HEENT: NC/AT, MMM
Respiratory: CTAB
Cardiovascular: RRR, normal S1S2, no JVD
Gastrointestinal: Abd soft, NT, ND, +BS
"""

    resp = await safe_ai_call(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.text.strip()},
        ],
        temperature=0.0,
        max_completion_tokens=800,
    )

    return SummaryResponse(summary=resp.choices[0].message.content.strip())

# ─────────────────────────────────────────────────────────────
# FOLLOW-UP RECOMMENDATION GENERATOR
# ─────────────────────────────────────────────────────────────

@v1.post("/generate-follow-up-recommendations", response_model=SummaryResponse)
async def generate_follow_up_recommendations(request: GenerateAPV2Request):
    """
    Produces bullet-style follow-up recommendations based on current and past notes.
    Reuses same schema as GenerateAPV2Request for convenience.
    """
    sys_prompt = """
    You are a consultant preparing follow-up recommendations after reviewing a patient case.
    Create concise, actionable next steps in bullet form (•).
    Include responsible teams, timelines, and any repeat studies or follow-ups.
    """
    user_prompt = f"CURRENT NOTE:\n{request.current_note_context}\n\nPAST NOTES:\n{request.past_notes_context or 'N/A'}"
    resp = await safe_ai_call(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_completion_tokens=600,
    )
    return SummaryResponse(summary=resp.choices[0].message.content.strip())

# ─── Discharge Summary ───────────────────────────────────────────────────────
@v1.post("/generate-discharge-summary", response_model=SummaryResponse)
async def generate_discharge_summary(request: NotesPayload):
    """
    Generate a polished, structured discharge summary from a list of recent notes.
    """
    notes = request.notes
    if not notes:
        raise HTTPException(status_code=400, detail="Need at least one note")

    formatted = "\n\n".join(
        f"🗓 {n.timestamp.strftime('%Y-%m-%d %H:%M')} — {n.type}\n"
        f"HPI: {n.hpi or n.subjective or 'N/A'}\n"
        f"Exam: {n.exam or 'N/A'}\n"
        f"Objective: {n.objective or 'N/A'}\n"
        f"A&P: {n.ap or 'N/A'}\n"
        f"MDM: {n.mdm or 'N/A'}"
        for n in notes
    )

    system_prompt = """
You are a hospitalist composing a professional **Discharge Summary** for inclusion in the medical record.
Your goal is to clearly communicate the patient's hospital course and discharge plan to the next provider.

Structure the output into the following sections:

1. **Admission Summary** — 2–4 sentences summarizing the reason for admission and initial presentation.
2. **Hospital Course** — concise overview of key problems, interventions, treatments, and response to therapy.
3. **Discharge Medications / Instructions** — summarize new medications, changes, or important discontinuations.
4. **Follow-Up / Pending Results** — list specific follow-up plans, appointments, and pending tests.

Requirements:
- Use concise, factual, and clinically accurate language.
- Do not include filler phrases or disclaimers.
- Avoid redundant restatements of vitals or labs unless clinically relevant.
- Write in complete sentences suitable for a discharge document.
""".strip()

    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted}
        ],
        temperature=0.4
    )
    summary = resp.choices[0].message.content.strip()
    return SummaryResponse(summary=summary)


# ─── Hospital Course ──────────────────────────────────────────────────────────
@v1.post("/generate-hospital-summary", response_model=SummaryResponse)
async def generate_hospital_summary(request: NotesPayload):
    """
    Generate a polished, structured hospital course summary from a list of notes.
    """
    notes = request.notes
    if not notes:
        raise HTTPException(status_code=400, detail="Need at least one note")

    formatted = "\n\n".join(
        f"🗓 {n.timestamp.strftime('%Y-%m-%d %H:%M')} — {n.type}\n"
        f"HPI: {n.hpi or n.subjective or 'N/A'}\n"
        f"Exam: {n.exam or 'N/A'}\n"
        f"Objective: {n.objective or 'N/A'}\n"
        f"A&P: {n.ap or 'N/A'}"
        for n in notes
    )

    system_prompt = """
You are a hospitalist summarizing a patient's entire hospital stay.
Write a polished, professional, problem-based hospital course in a medically concise format.

Your output must include:

1. **Overview** – 2–4 sentences summarizing admission reason, course highlights, and trajectory.
2. **Hospital Course (Problem-Based)** – concise bullet points by major problems or organ systems.
3. **Pending Issues / Follow-Up** – brief items for outpatient care or pending results.

Requirements:
- Use concise, factual, clinically accurate language.
- Avoid redundancy and conversational tone.
- Do not invent details.
- Output should be ready for direct insertion into the record.
""".strip()

    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted}
        ],
        temperature=0.4
    )
    summary = resp.choices[0].message.content.strip()
    return SummaryResponse(summary=summary)

# ─── MDM Generator───────────────────────────────────────────────────────

@v1.post("/generate-mdm", response_model=SummaryResponse)
async def generate_mdm(request: GenerateMDMRequest):
    """
    Generates a true Medical Decision Making (MDM) section.
    This version focuses on the "why" behind the diagnosis, not the "what" of the plan.
    """
    if not any([request.hpi, request.exam, request.objective, request.assessment]):
        raise HTTPException(status_code=400, detail="Not enough context to generate MDM.")

    system_prompt = """
You are an expert hospitalist writing ONLY the MEDICAL DECISION MAKING (MDM) section of a note. Your entire output will be a single, well-structured paragraph.

**CRITICAL INSTRUCTIONS:**
1.  **NO PLAN:** Do NOT include any plan items, orders, medications, or treatments. Focus exclusively on the thought process.
2.  **STRUCTURE:** Structure your paragraph around the differential diagnosis, data interpretation, and risk assessment.
3.  **DIFFERENTIAL DIAGNOSIS:** Start by stating the primary diagnosis and the key evidence supporting it. List 2-3 other important differential diagnoses that were considered and briefly state why they are less likely.
4.  **DATA INTERPRETATION:** Discuss the key labs, imaging, or exam findings that informed your decision-making. (e.g., "The elevated troponin and EKG changes were concerning for ACS," "The negative D-dimer made PE less likely.").
5.  **RISK & COMPLEXITY:** Conclude with an assessment of the patient's overall risk (e.g., "Overall, this is a case of high medical complexity given the patient's comorbidities and acute presentation.").

Your entire response must be a single block of text.
"""

    clinical_context = f"""
    HPI: {request.hpi or 'Not provided.'}
    EXAM: {request.exam or 'Not provided.'}
    OBJECTIVE DATA (Labs/Imaging): {request.objective or 'Not provided.'}
    ASSESSMENT: {request.assessment or 'Not provided.'}
    """
    
    resp = await safe_ai_call(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": clinical_context},
        ],
        temperature=0.3,
        max_tokens=800
    )

    return SummaryResponse(summary=resp.choices[0].message.content.strip())

# --- Improving Text ---
@v1.post("/generate-text", response_model=SummaryResponse)
async def generate_text(payload: dict):
    """
    Context-aware text refinement endpoint.
    Dynamically adjusts tone and structure depending on note type (hospital course, discharge, progress, etc.)
    """
    text = payload.get("prompt", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing prompt")

    # Detect likely context
    lower_text = text.lower()

    if "hospital course" in lower_text or "admission" in lower_text:
        context_role = "You are a hospitalist summarizing an inpatient course."
        style_tone = (
            "Refine this hospital course summary to sound professional, concise, "
            "and medically precise. Maintain a chronological flow and problem-based logic. "
            "Avoid redundancy and filler phrases."
        )
    elif "discharge" in lower_text:
        context_role = "You are a hospitalist preparing a discharge summary."
        style_tone = (
            "Refine this discharge summary to ensure clarity and professionalism. "
            "Summarize key hospital events, condition at discharge, and follow-up plans succinctly."
        )
    elif "progress" in lower_text or "follow-up" in lower_text:
        context_role = "You are an inpatient clinician writing a daily progress note."
        style_tone = (
            "Refine the following progress note for medical precision and concise structure. "
            "Emphasize interval updates, active problems, and plans in a problem-based format."
        )
    elif "consult" in lower_text:
        context_role = "You are a consulting physician."
        style_tone = (
            "Refine this consult note to maintain professional tone, include a clear assessment and recommendation section, "
            "and remove redundancies or conversational phrasing."
        )
    else:
        context_role = "You are a medical writer improving general clinical documentation."
        style_tone = (
            "Refine this text to improve grammar, flow, and conciseness while preserving all clinical meaning. "
            "Avoid hallucination or added diagnoses."
        )

    # Combine prompts
    system_prompt = f"{context_role} {style_tone}"

    # Generate refined text
    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.45
    )

    return SummaryResponse(summary=resp.choices[0].message.content.strip())

# ─────────────────────────────────────────────────────────────
# DRUG SEARCH (RxNav)
# ─────────────────────────────────────────────────────────────

@v1.get("/search-drugs", response_model=DrugSearchResponse)
async def search_drugs(query: str):
    if not query.strip():
        return DrugSearchResponse(results=[])
    encoded = quote(query)
    url = f"https://rxnav.nlm.nih.gov/REST/drugs.json?name={encoded}"
    async with httpx.AsyncClient(timeout=5.0) as client_http:
        resp = await client_http.get(url)
        resp.raise_for_status()
        data = resp.json()
    results = [
        {"id": c.get("rxcui", ""), "name": c.get("name", ""), "isControlled": False}
        for g in data.get("drugGroup", {}).get("conceptGroup", [])
        for c in g.get("conceptProperties", [])
    ]
    return DrugSearchResponse(results=results)

# ─── MEDICATION NORMALIZER ───────────────────────────────────────────────────────

@v1.post("/normalize-medications", response_model=SummaryResponse)
async def normalize_medications(request: TranscriptionResponse):
    """
    Cleans and normalizes medication lists by correcting common transcription errors,
    expanding abbreviations, and confirming names against known drugs.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(400, "Missing medication text")

    system_prompt = """
You are a clinical pharmacist reviewing a transcribed medication list.
Correct any likely transcription or spelling errors in medication names.
Preserve dosages, routes, and frequencies. 
If uncertain, return the most likely correct medication name (e.g., 'rosatin' → 'losartan').
Output a clean, human-readable medication list (plain text, not JSON).
"""

    resp = await safe_ai_call(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.1,
        max_completion_tokens=400,
    )

    return SummaryResponse(summary=resp.choices[0].message.content.strip())


# ─────────────────────────────────────────────────────────────
# SUBSCRIPTION STATUS ENDPOINT (Web + iOS shared)
# ─────────────────────────────────────────────────────────────

@v1.get("/subscription-status/{uid}")
async def get_subscription_status(uid: str):
    """
    Returns whether the user currently has full Pro access.
    Logic:
      - If subscription doc exists and isPro == true → Pro access
      - Else if today < trialEnds → still in free trial → Pro access
      - Else → locked
    """

    doc = db.collection("subscriptions").document(uid).get()

    # Default locked state
    status = {
        "isPro": False,
        "trialActive": False,
        "hasProAccess": False,
        "trialEnds": None
    }

    if not doc.exists:
        # No subscription record → treat as no access
        return status

    data = doc.to_dict()

    is_pro = data.get("isPro", False)
    trial_ends = data.get("trialEnds")
    trial_active = False

    # Check trial window
    if trial_ends:
        try:
            trial_date = datetime.fromisoformat(trial_ends.replace("Z", "+00:00"))
            trial_active = datetime.utcnow() < trial_date
        except Exception:
            trial_active = False

    # Final logic
    has_access = is_pro or trial_active

    status.update({
        "isPro": is_pro,
        "trialActive": trial_active,
        "hasProAccess": has_access,
        "trialEnds": trial_ends
    })

    return status


# -------------------------------------------------
# v1 ROUTES
# -------------------------------------------------
@v1.get("/health")
async def health_v1():
    return {"status": "ok", "version": "v1"}

# (all other v1 endpoints here)

# -------------------------------------------------
# v2 ROUTES
# -------------------------------------------------
@v2.get("/health")
async def health_v2():
    return {"status": "ok", "version": "v2"}

# (all other v2 endpoints here)


# -------------------------------------------------
# MOUNT ROUTERS (must be last)
# -------------------------------------------------

# Versioned APIs
app.include_router(v1)
app.include_router(v2)

# Feature routers (already versioned)
app.include_router(subscriptions_router, prefix="/v1")

