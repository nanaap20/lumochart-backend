import os
import io
import time
import json
import re
import asyncio
import re

from datetime import datetime
from typing import List, Optional

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    BackgroundTasks
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import storage
from pydub import AudioSegment
import httpx
from urllib.parse import quote

from enum import Enum
import base64

# — Load environment variables
load_dotenv()

# — Initialize Firebase / Firestore bucket
firebase_admin.initialize_app(options={
    "storageBucket": os.getenv("FIREBASE_BUCKET", "lumochart.firebasestorage.app")
})
bucket = storage.bucket()

# — Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# — FastAPI app & CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# ─── Health-check / Root Endpoint ─────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "LumoChart backend is running 🎉"}

# ─── In-memory transcription job store ────────────────────────────────────
transcription_jobs: dict[str, dict] = {}

def log_event(level: str, event: str, **details):
    """
    Central logging utility. Writes JSON to stdout—ideal for structured logs.
    """
    print(json.dumps({
        "severity": level,
        "event": event,
        "timestamp": time.time(),
        "details": details
    }))

# ─── Pydantic Models ───────────────────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str

class TextOrTranscript(BaseModel):
    text: Optional[str] = None
    transcript: Optional[str] = None

class EnhanceTextRequest(BaseModel):
    original_text: str
    new_dictation: str

class ContextualAPRequest(BaseModel):
    current_note_context: str
    past_notes_context: str

class TranscriptionRequest(BaseModel):
    job_id: str
    file_path: str

class TranscriptionResponse(BaseModel):
    text: str

class PromptRequest(BaseModel):
    prompt: str

class Note(BaseModel):
    id: Optional[str] = None
    patientID: str
    ownerId: str
    timestamp: datetime = datetime.now()
    type: str
    summary: Optional[str] = None
    hpi: Optional[str] = None
    ros: Optional[str] = None
    exam: Optional[str] = None
    ap: Optional[str] = None
    mdm: Optional[str] = None
    pastMedicalHistory: Optional[str] = None
    medications: Optional[str] = None
    allergies: Optional[str] = None
    pastSurgicalHistory: Optional[str] = None
    socialHistory: Optional[str] = None
    familyHistory: Optional[str] = None
    subjective: Optional[str] = None
    objective: Optional[str] = None
    consultQuestion: Optional[str] = None
    intervalHistory: Optional[str] = None
    followUp: Optional[str] = None
    medicationAdherence: Optional[str] = None
    lifestyleChanges: Optional[str] = None
    functionalStatus: Optional[str] = None
    patientConcerns: Optional[str] = None
    disposition: Optional[str] = None
    billingSuggestion: Optional[str] = None

    class Config:
        # Allow extra fields without error
        extra = "ignore"

class EDHistoryResponse(BaseModel):
    hpi: str
    review_of_systems: Optional[str] = ""

class EDInitialCaptureResponse(BaseModel):
    transcript: str
    history: EDHistoryResponse

class EDSummaryResponse(BaseModel):
    assessment_and_plan: str
    mdm: str
    disposition_recommendation: Optional[str] = ""
    billing_level_suggestion: Optional[str]   = ""

class SubjectiveResponse(BaseModel):
    hpi: str
    past_medical_history: Optional[str] = None
    medications: Optional[str] = None
    allergies: Optional[str] = None
    past_surgical_history: Optional[str] = None
    social_history: Optional[str] = None
    family_history: Optional[str] = None
    review_of_systems: Optional[str] = None

class ProgressNoteResponse(BaseModel):
    subjective: Optional[str] = ""
    interval_history: Optional[str] = ""
    new_symptoms: Optional[str] = ""
    side_effects: Optional[str] = ""
    objective: Optional[str] = ""
    prior_problem_progress: Optional[str] = ""
    medication_changes: Optional[str] = ""
    ap: Optional[str] = ""
    updated_plan: Optional[str] = ""


class InitialConsultNoteResponse(BaseModel):
    consult_question: Optional[str] = ""
    hpi: Optional[str] = ""
    exam: Optional[str] = ""
    ap: Optional[str] = ""

class FollowUpConsultNoteResponse(BaseModel):
    consult_question: Optional[str] = ""
    interval_history: Optional[str] = ""
    exam: Optional[str] = ""
    assessment: Optional[str] = ""
    plan: Optional[str] = ""
    follow_up: Optional[str] = ""

class InitialClinicNoteResponse(BaseModel):
    hpi: str
    review_of_systems: Optional[str] = ""
    past_medical_history: Optional[str] = ""
    past_surgical_history: Optional[str] = ""
    family_history: Optional[str] = ""
    medications: Optional[str] = ""
    allergies: Optional[str] = ""
    social_history: Optional[str] = ""
    exam: Optional[str] = ""
    assessment_plan: Optional[str] = ""


class FollowUpClinicNoteResponse(BaseModel):
    interval_history: str
    review_of_systems: Optional[str] = ""
    prior_problem_progress: Optional[str] = ""
    new_symptoms: Optional[str] = ""
    new_concerns: Optional[str] = ""
    medication_changes: Optional[str] = ""
    side_effects: Optional[str] = ""
    medication_adherence: Optional[str] = ""
    updated_plan: str

class TranscriptWithContext(BaseModel):
    transcript: str
    prior_notes: List[dict]

class SummaryResponse(BaseModel):
    summary: str

class MDMResponse(BaseModel):
    mdm: str

class SummaryNote(BaseModel):
    timestamp: datetime
    type: Optional[str] = None
    hpi: Optional[str] = None
    subjective: Optional[str] = None
    exam: Optional[str] = None
    objective: Optional[str] = None
    ap: Optional[str] = None
    mdm: Optional[str] = None

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

class APOptions(BaseModel):
    insist_specificity: bool = True           # force concrete actions (drug, dose, route, frequency, duration)
    include_dosing: bool = True               # include dose/route/frequency defaults when safe
    include_monitoring: bool = True           # include monitoring/labs, stop criteria
    include_disposition: bool = True          # include level of care/admit vs discharge when applicable
    include_consult_triggers: bool = True     # “call X if Y” triggers
    cite_guideline_hints: bool = False        # brief parenthetical guideline hints (no URLs)
    avoid_handwaving: bool = True             # remove generic “consider/if needed” fluff
    force_commitment: bool = True             # choose a plan and state why

class GenerateAPV2Request(BaseModel):
    context: str                               # "admission" | "progress" | "consult" | "outpatient" | "ed"
    current_note_context: str                  # what you currently pass in (assembled text)
    past_notes_context: Optional[str] = ""     # optional past notes blob
    consult_question: Optional[str] = None     # helps force relevance
    patient_factors: Optional[str] = None      # CKD, pregnancy, QT risk, etc.
    options: APOptions = APOptions()

class IntervalHistoryRequest(BaseModel):
    mode: str                                  # "consult_followup" | "progress"
    transcript: str
    consult_question: Optional[str] = None
    prior_notes: Optional[List[dict]] = []

class MergeNoteRequest(BaseModel):
    note_type: str                   # admission, progress, outpatient, consult, ed
    existing_note: dict              # existing structured note JSON
    new_text: str                    # new transcript or snippet
    past_notes: Optional[List[dict]] = []   # prior notes for continuity (esp. progress/consult)

class Drug(BaseModel):
    id: str
    name: str
    isControlled: bool = False

class DrugSearchResponse(BaseModel):
    results: List[dict]


class EnhanceTextResponse(BaseModel):
    summary: str


class UpdatedPlanRequest(BaseModel):
    new_ap: str
    prior_ap: Optional[str] = ""


# ───────── End of Part 1 ─────────

# ─── Audio Helpers ─────────────────────────────────────────────────────────────

def get_audio_duration(input_bytes: bytes) -> float:
    """
    Return the duration (in seconds) of raw audio bytes using ffprobe.
    """
    from tempfile import NamedTemporaryFile
    import subprocess
    import os

    with NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(input_bytes)
        tmp_path = tmp.name

    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        tmp_path
    ])
    os.remove(tmp_path)
    return float(out)

def split_audio_into_files(raw_bytes: bytes, chunk_seconds: int) -> list[bytes]:
    """
    Split raw audio bytes into chunks of up to chunk_seconds.
    Returns list of raw bytes for each chunk.
    """
    from pydub import AudioSegment
    import io

    seg = AudioSegment.from_file(io.BytesIO(raw_bytes))
    if len(seg) <= chunk_seconds * 1000:
        return [raw_bytes]

    chunks: list[bytes] = []
    for i in range(0, len(seg), chunk_seconds * 1000):
        piece = seg[i : i + chunk_seconds * 1000]
        buf = io.BytesIO()
        piece.export(buf, format="mp3")
        chunks.append(buf.getvalue())

    return chunks


# ─── Speaker-Tagging Preprocessor ─────────────────────────────────────────────

def tag_patient_speech(raw_transcript: str) -> str:
    """
    Prepend “Patient:” or “Clinician:” tags to each line.
    Simple heuristic: lines ending in '?' are clinician; others patient.
    """
    lines = raw_transcript.splitlines()
    tagged_lines = []
    for line in lines:
        txt = line.strip()
        if not txt:
            continue
        if txt.endswith("?"):
            tagged_lines.append(f"Clinician: {txt}")
        else:
            tagged_lines.append(f"Patient: {txt}")

    header = (
        "Below is a doctor–patient conversation. "
        "Lines prefixed “Patient:” contain ONLY what the patient said.\n\n"
    )
    return header + "\n".join(tagged_lines)


# ─── Central AI-Call Wrapper ─────────────────────────────────────────────────

async def ai_completion_or_fail(
    model: str,
    messages: list[dict],
    tools: list[dict] = None,
    tool_choice: dict = None,
    temperature: float = 0.0,
    max_retries: int = 2
):
    """
    Call OpenAI chat.completions.create with retries, backoff, and logging.
    Raises HTTPException(502) if the service remains unavailable.
    """
    attempt = 0
    while attempt <= max_retries:
        try:
            return await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature
            )
        except Exception as e:
            log_event("WARNING", "ai_call_error", attempt=attempt, error=str(e))
            attempt += 1
            await asyncio.sleep(2 ** attempt)

    raise HTTPException(502, detail="AI service unavailable after retries")


# ─── Robust JSON-Parsing Helper ────────────────────────────────────────────────

def parse_ai_json(raw: str, endpoint: str) -> dict:
    """
    Extract JSON object from raw AI output, strip markdown/text, parse to dict.
    Raises HTTPException(500) on failure and logs the raw output.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        log_event("ERROR", f"{endpoint}_invalid_json", raw=raw)
        raise HTTPException(500, detail=f"{endpoint} failed to return valid JSON:\n{raw}")

    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        log_event("ERROR", f"{endpoint}_json_decode_error", raw=raw, error=str(e))
        raise HTTPException(500, detail=f"{endpoint} JSON parse error:\n{raw}")

# ─── Prompt & Schema Templates ─────────────────────────────────────────────────

# Fields for H&P extraction (used by format_extract_prompt or simple calls)
HNP_FIELDS = [
    "hpi",
    "past_medical_history",
    "medications",
    "allergies",
    "past_surgical_history",
    "social_history",
    "family_history",
    "review_of_systems"
]

def format_extract_prompt(role: str, fields: list[str]) -> str:
    """
    Build a system prompt to extract ONLY the given fields into JSON.
    """
    bullets = "\n".join(f"- {f}" for f in fields)
    return (
        f"You are a {role}. Extract ONLY the following into JSON:\n"
        f"{bullets}\n\n"
        "If not mentioned, return an empty string for that field.\n"
        "Output ONLY the JSON object."
    )

# ─── JSON schema definitions for function-call endpoints ──────────────────────

EXTRACT_HNP_FN = {
    "name": "extract_hnp",
    "description": "Extract H&P fields from a doctor–patient transcript",
    "parameters": {
        "type": "object",
        "properties": {
            "hpi":                   { "type": "string", "description": "History of present illness" },
            "past_medical_history":  { "type": "string" },
            "medications":           { "type": "string" },
            "allergies":             { "type": "string" },
            "past_surgical_history": { "type": "string" },
            "social_history":        { "type": "string" },
            "family_history":        { "type": "string" },
            "review_of_systems":     { "type": "string" }
        },
        "required": ["hpi"]
    }
}

PROGRESS_NOTE_FN = {
    "name": "extract_progress_note",
    "description": "Extract structured progress note fields from a transcript.",
    "parameters": {
        "type": "object",
        "properties": {
            "subjective": {"type": "string"},
            "interval_history": {"type": "string"},
            "new_symptoms": {"type": "string"},
            "side_effects": {"type": "string"},
            "objective": {"type": "string"},
            "prior_problem_progress": {"type": "string"},
            "medication_changes": {"type": "string"},
            "ap": {"type": "string"},
            "updated_plan": {"type": "string"}
        },
        "required": ["subjective", "interval_history", "objective", "ap", "updated_plan"]
    }
}


EXTRACT_INITIAL_CLINIC_FN = {
    "name": "extract_initial_clinic_note",
    "description": "Extract structured clinical content from an initial outpatient visit transcript.",
    "parameters": {
        "type": "object",
        "properties": {
            "hpi":                   {"type": "string", "description": "Reason for visit / HPI"},
            "review_of_systems":     {"type": "string", "description": "System-based ROS (General, HEENT, CV, Resp, GI, GU, MSK, Neuro, Skin, Psych) using 'denies/endorses/reports' wording."},
            "past_medical_history":  {"type": "string"},
            "past_surgical_history": {"type": "string"},
            "family_history":        {"type": "string"},
            "medications":           {"type": "string"},
            "allergies":             {"type": "string"},
            "social_history":        {"type": "string"},
            "exam":                  {"type": "string"},
            "assessment_plan":       {"type": "string"}
        },
        "required": ["hpi", "review_of_systems"]
    }
}



FOLLOWUP_CLINIC_FN = {
    "name": "extract_clinic_followup_note",
    "description": "Extract follow-up visit fields using transcript and prior clinical context.",
    "parameters": {
        "type": "object",
        "properties": {
            "interval_history":              {"type": "string"},
            "review_of_systems":             {"type": "string"},
            "prior_problem_progress":        {"type": "string"},
            "new_symptoms":                  {"type": "string"},
            "new_concerns":                  {"type": "string"},
            "medication_changes":            {"type": "string"},
            "side_effects":                  {"type": "string"},
            "medication_adherence":          {"type": "string"},
            "updated_plan":                  {"type": "string"}
        },
        "required": [
            "interval_history",
            "review_of_systems",
            "prior_problem_progress",
            "new_symptoms",
            "new_concerns",
            "medication_changes",
            "side_effects",
            "medication_adherence",
            "updated_plan"
        ]
    }
}


INITIAL_CONSULT_FN = {
    "name": "extract_initial_consult_note",
    "description": "Extract initial inpatient consult fields from transcript.",
    "parameters": {
        "type": "object",
        "properties": {
            "consult_question": {"type": "string"},
            "hpi":              {"type": "string"},
            "exam":             {"type": "string"},
            "ap":               {"type": "string"}
        },
        "required": ["consult_question", "hpi", "exam", "ap"]
    }
}

FOLLOWUP_CONSULT_FN = {
    "name": "extract_consult_followup_note",
    "description": "Extract follow-up consult fields from transcript.",
    "parameters": {
        "type": "object",
        "properties": {
            "consult_question": {"type": "string"},
            "interval_history": {"type": "string"},
            "exam": {"type": "string"},
            "assessment": {"type": "string"},
            "plan": {"type": "string"},
            "follow_up": {"type": "string"}
        },
        "required": ["consult_question", "interval_history", "exam", "assessment", "plan", "follow_up"]
    }
}


EXTRACT_ED_HISTORY_FN = {
    "name": "extract_ed_history",
    "description": "Extract hpi and review_of_systems from an ED transcript",
    "parameters": {
        "type": "object",
        "properties": {
            "hpi":               {"type": "string"},
            "review_of_systems": {"type": "string"}
        },
        "required": ["hpi"]
    }
}

EXTRACT_ED_SUMMARY_FN = {
    "name": "extract_ed_summary",
    "description": "Extracts A&P, MDM, Disposition, and Billing from an ED transcript.",
    "parameters": {
        "type": "object",
        "properties": {
            "assessment_and_plan": {
                "type": "string",
                "description": "A concise, problem-based Assessment & Plan for the ED physician."
            },
            "mdm": {
                "type": "string",
                "description": "Medical Decision Making justifying the workup and final disposition."
            },
            "disposition_recommendation": {
                "type": "string",
                "description": "Clear disposition recommendation (e.g., admit, discharge, observe) with rationale."
            },
            "billing_level_suggestion": {
                "type": "string",
                "description": "Suggested E/M billing level (e.g., 99284) with a one-sentence justification."
            }
        },
        "required": ["assessment_and_plan", "mdm", "disposition_recommendation"]
    }
}

INTERVAL_HISTORY_FN = {
    "name": "extract_interval_history",
    "description": "Extract concise interval history emphasizing changes since last note and relevance to consult question if present.",
    "parameters": {
        "type": "object",
        "properties": {
            "interval_history": {"type": "string", "description": "Clear, focused interval history since last note."},
            "new_symptoms": {"type": "string"},
            "medication_changes": {"type": "string"},
            "adverse_effects": {"type": "string"},
            "prior_problem_progress": {"type": "string"},
            "patient_concerns": {"type": "string"}
        },
        "required": ["interval_history"]
    }
}


# ───────── End of Part 2 ─────────


from fastapi import BackgroundTasks, File, UploadFile, Form

# Background transcription worker
async def process_transcription_in_background(job_id: str, contents: bytes, filename: str):
    def update_status(status: str, message: str = "", transcript: str | None = None):
        transcription_jobs[job_id] = {
            "status": status,
            "transcript": transcript,
            "message": message
        }

    update_status("processing", "Preparing audio…")

    try:
        buf = io.BytesIO(contents)
        buf.name = filename

        update_status("processing", "Sending to Whisper model…")
        result = await client.audio.transcriptions.create(
            model="whisper-1",
            file=buf
        )

        transcript = result.text.strip()
        update_status("complete", "Transcription successful.", transcript=transcript)

    except Exception as e:
        update_status("failed", f"Transcription failed: {str(e)}")


# Start a non-blocking transcription job (direct upload)
@app.post("/start-transcription-job")
async def start_transcription_job(
    background_tasks: BackgroundTasks,
    job_id: str = Form(...),
    file: UploadFile = File(...)
):
    contents = await file.read()
    transcription_jobs[job_id] = {"status": "processing", "transcript": None}
    background_tasks.add_task(process_transcription_in_background, job_id, contents, file.filename)
    return {"status": "processing", "job_id": job_id}


# Check transcription job status
@app.get("/get-job-status/{job_id}")
async def get_job_status(job_id: str):
    job = transcription_jobs.get(job_id)
    if not job:
        return {"status": "unknown", "message": "Job not found", "transcript": None}
    return {
        "status": job.get("status", "unknown"),
        "message": job.get("message", ""),
        "transcript": job.get("transcript")
    }


# Synchronous (immediate) transcription endpoint
@app.post("/transcribe-objective", response_model=TranscriptionResponse)
async def transcribe_objective(file: UploadFile = File(...)):
    contents = await file.read()
    buffer = io.BytesIO(contents)
    buffer.name = file.filename
    try:
        result = await client.audio.transcriptions.create(model="whisper-1", file=buffer)
        return TranscriptionResponse(text=result.text)
    except Exception as e:
        log_event("ERROR", "transcribe_objective_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    
# ───────── End of Part 3 ─────────


# ───────── Generate Note From Capture ─────────
@app.post("/generate-note-from-capture")
async def generate_note_from_capture(request: SmartCaptureRequest):
    """
    Receives a multi-modal payload (text, audio transcripts, images),
    consolidates them into a single text context, and then generates a
    structured clinical note based on the requested note_type.
    """
    consolidated_context = []
    
    # Step 1: Process all inputs and consolidate them into text
    for capture_input in request.inputs:
        if capture_input.type in [CaptureInputType.audio_transcript, CaptureInputType.text_snippet]:
            consolidated_context.append(capture_input.content)
        
        elif capture_input.type == CaptureInputType.image_base64:
            try:
                vision_response = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image for a clinical note. If it contains text like a lab report or EKG, extract it accurately."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{capture_input.content}"}}
                        ]
                    }],
                    max_tokens=500
                )
                image_description = vision_response.choices[0].message.content
                consolidated_context.append(f"Image Content:\n{image_description}")
            except Exception as e:
                log_event("WARNING", "vision_api_call_failed", error=str(e))
                consolidated_context.append("[Image processing failed]")

    full_context = "\n\n".join(consolidated_context)

    note_type = request.note_type
    
    # --- THIS IS THE UPDATED LOGIC ---
    if note_type == "ed":
        # For ED notes, we do the history extraction here and package it with the transcript.
        history_resp = await ai_completion_or_fail(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an ED scribe. Extract only HPI and ROS."},
                {"role": "user", "content": full_context}
            ],
            tools=[{"type": "function", "function": EXTRACT_ED_HISTORY_FN}],
            tool_choice={"type": "function", "function": {"name": "extract_ed_history"}}
        )
        history_args = json.loads(history_resp.choices[0].message.tool_calls[0].function.arguments)
        history_data = EDHistoryResponse(**history_args)
        
        return EDInitialCaptureResponse(transcript=full_context, history=history_data)
    
    # For all other note types, we select the appropriate tool and response model.
    note_configs = {
        "admission": (EXTRACT_INITIAL_CLINIC_FN, "extract_initial_clinic_note", InitialClinicNoteResponse),
        "outpatient": (EXTRACT_INITIAL_CLINIC_FN, "extract_initial_clinic_note", InitialClinicNoteResponse),
        "progress": (PROGRESS_NOTE_FN, "extract_progress_note", ProgressNoteResponse),
        "consult": (INITIAL_CONSULT_FN, "extract_initial_consult_note", InitialConsultNoteResponse)
    }

    if note_type not in note_configs:
        raise HTTPException(status_code=400, detail=f"Unknown note_type: {note_type}")

    tool_fn, tool_choice_name, response_model = note_configs[note_type]

    system_prompt = f"You are a medical scribe creating a {note_type} note. Based on the provided context, extract the required fields using the provided tool."
    
    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_context},
        ],
        tools=[{"type": "function", "function": tool_fn}],
        tool_choice={"type": "function", "function": {"name": tool_choice_name}},
        temperature=0.1
    )

    tool_calls = resp.choices[0].message.tool_calls
    if not tool_calls:
        raise HTTPException(status_code=500, detail=f"Note generation failed for note_type {note_type}")

    args = json.loads(tool_calls[0].function.arguments)
    return response_model(**args)

# ─── H&P Extraction ──────────────────────────────────────────────────────────
@app.post("/process-hnp-from-text", response_model=SubjectiveResponse)
async def process_hnp_from_text(request: TextRequest):
    tagged = tag_patient_speech(request.text)

    system_prompt = """
    You are a medical scribe for inpatient documentation. Extract clinically relevant subjective information from the transcript and return clean plain-text values for each of these fields in your function call:  
    • hpi  
    • past_medical_history  
    • medications  
    • allergies  
    • past_surgical_history  
    • social_history  
    • family_history  
    • review_of_systems  

    Always output **only** the text for each field—no JSON keys, no headers, no bullets or extra formatting. The JSON schema will wrap your response. If a field isn’t present in the transcript, return an empty string (or “No known allergies” for allergies if explicitly denied).

    **HPI**  
    - Begin with **age, sex, key PMH, and chief complaint in the same sentence.**  
      Example: “77 y/o M with PMH of ESRD on HD who presents today with complaints of shortness of breath and cough.”  
    - After this opener, continue with a **chronological narrative**: onset, evolution, treatments tried, collateral history, ED findings, etc.  
    - Use smooth transitions (“Initially…”, “Over the next two days…”, “On arrival…”).  
    - Conclude with relevant ED vitals, labs, or imaging.  
    - Keep as one concise flowing paragraph (multiple sentences allowed).

    **Medications**  
    - Flatten into an inline, semicolon-separated list.  
    - If none are mentioned, return an empty string.

    **Allergies**  
    - List all allergies inline (e.g., “Penicillin: rash; NKDA”).  
    - If explicitly denied, return “No known allergies.”  
    - If unmentioned, return an empty string.

    **Review of Systems**  
    - Group by body system with brief sentences (e.g., “General: denies fever or chills. Cardiovascular: denies chest pain or palpitations. Respiratory: endorses cough and SOB.”).  
    - Use “denies,” “endorses,” or “reports” for each finding.  
    - If no ROS data, return an empty string.

    **Social, Surgical, and Family History**  
    - Write each as a short factual sentence or two (e.g., “Lives with wife; former smoker, quit 2010.”).  
    - Avoid excessive detail; focus on clinically relevant facts.  
    - If unmentioned, return an empty string.
    """

    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": tagged}
        ],
        tools=[{"type": "function", "function": EXTRACT_HNP_FN}],
        tool_choice={"type": "function", "function": {"name": "extract_hnp"}},
        temperature=0.0
    )

    tool_calls = resp.choices[0].message.tool_calls or []
    args = json.loads(tool_calls[0].function.arguments)
    return SubjectiveResponse(**args)




# ─── Inpatient Progress Note Extraction ─────────────────────────────────────
@app.post("/extract-progress-note", response_model=ProgressNoteResponse)
async def extract_progress_note(request: TranscriptWithContext):
    context_blob = f"""Transcript:\n{request.transcript}\n\nPrior Notes:\n{request.prior_notes}"""

    system_prompt = """
    You are a hospital provider generating a daily inpatient progress note.

    The note should reflect the patient’s clinical status, overnight events, new issues, medication changes, and diagnostic evolution. Extract each field below from the transcript:

    - subjective: Patient-reported symptoms, feelings, concerns
    - interval_history: Changes since prior note—overnight events, new findings
    - new_symptoms: Anything newly reported today
    - side_effects: Medication side effects or adverse reactions
    - objective: Physical exam findings, vital signs, labs, imaging
    - prior_problem_progress: Updates on known diagnoses or conditions
    - medication_changes: Dosing changes, new starts/stops
    - ap: Clinical interpretation and plan per issue
    - updated_plan: Any specific instructions or disposition updates

    Stay concise and clinically clear. Format each field as a stand-alone paragraph.
    """

    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context_blob}
        ],
        tools=[{"type":"function","function": PROGRESS_NOTE_FN}],
        tool_choice={"type":"function","function":{"name": "extract_progress_note"}},
        temperature=0.0
    )

    args = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
    return ProgressNoteResponse(**args)



# ─── Initial Clinic Visit Extraction ────────────────────────────────────────
@app.post("/extract-initial-clinic-note", response_model=InitialClinicNoteResponse)
async def extract_initial_clinic_note(request: TextOrTranscript):
    raw = (request.text or request.transcript or "").strip()
    if not raw:
        raise HTTPException(400, detail="Provide 'text' or 'transcript'")

    tagged = tag_patient_speech(raw)

    system_prompt = """
    You are a medical scribe documenting an initial outpatient visit. Extract clean plain text for each field.

    **HPI**  
    - Begin with **age, sex, key PMH, AND chief complaint in the same sentence.**  
      Example: “69 y/o F with PMH of HTN and DM who presents today with complaints of fatigue and shortness of breath.”  
    - If age/sex unknown, at least include PMH + chief complaint in the opener.  
    - After the opener, continue with a concise chronological narrative: onset, symptom evolution, treatments tried, collateral history, clinic findings.  
    - Conclude with relevant vitals, labs, or imaging if available.  
    - Keep phrasing clinical, concise, and in one flowing paragraph.

    **Past Medical History**  
    - Repeat chronic conditions explicitly, even if mentioned in HPI.

    **Medications**  
    - If adherence is mentioned but meds aren't listed, infer common agents (e.g., “lisinopril; metformin” for HTN + DM).  
    - Separate with semicolons; leave empty only if truly unknown.

    **Allergies**  
    - If not mentioned, default to “No known drug allergies.”

    **Review of Systems**  
    - Output system-based lines using concise sentences and “denies/endorses/reports.”  
    - Prefer: General, HEENT, Cardiovascular, Respiratory, GI, GU, MSK, Neuro, Skin, Psych.  
    - Include only systems referenced; if none, return empty string.

    **Social History**  
    - Include living situation, smoking, alcohol, drugs, occupation/support if mentioned.

    **Physical Exam**  
    - Summarize provider findings when available; else omit.

    **Assessment & Plan**  
    - Include diagnoses, rationale, and next steps if possible; else may be empty.

    Output via the provided tool. Do NOT include headers or extra formatting in values.
    """.strip()

    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": tagged}
        ],
        tools=[{"type": "function", "function": EXTRACT_INITIAL_CLINIC_FN}],
        tool_choice={"type": "function", "function": {"name": "extract_initial_clinic_note"}},
        temperature=0.0
    )

    tool_calls = resp.choices[0].message.tool_calls or []
    if not tool_calls:
        raise HTTPException(500, detail="Extractor returned no tool call")

    try:
        args = json.loads(tool_calls[0].function.arguments)
    except (json.JSONDecodeError, IndexError) as e:
        raise HTTPException(500, detail=f"Failed to parse extractor output: {e}")

    return InitialClinicNoteResponse(**args)



# ─── Clinic Follow-Up Extraction ────────────────────────────────────────
@app.post("/extract-clinic-followup-note", response_model=FollowUpClinicNoteResponse)
async def extract_clinic_followup_note(request: TranscriptWithContext):
    context_blob = f"""Prior Notes:\n{request.prior_notes}\n\nCurrent Transcript:\n{request.transcript}"""

    system_prompt = """
    You are a medical scribe documenting a follow-up outpatient visit. You are given the current transcript and prior clinical notes for this patient.

    Use prior notes to:
    – Identify active problems that need progress updates
    – Recognize resolved issues or missed follow-up items
    – Maintain longitudinal context across visits

    Transcript may include:
    – Interval history, new symptoms, updated medications, side effects, patient concerns

    Extract the following:
    – interval_history
    – review_of_systems
    – prior_problem_progress (status of known issues)
    – new_symptoms
    – new_concerns
    – medication_changes
    – side_effects
    – medication_adherence
    – updated_plan

    Be concise, clinically formatted, and ensure continuity of care. Write plain text only.
    """

    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": context_blob}
        ],
        tools=[{"type":"function","function": FOLLOWUP_CLINIC_FN}],
        tool_choice={"type":"function","function":{"name": "extract_clinic_followup_note"}},
        temperature=0.0
    )

    tool_calls = resp.choices[0].message.tool_calls or []
    args = json.loads(tool_calls[0].function.arguments)
    return FollowUpClinicNoteResponse(**args)



# ─── Initial Consult Extraction ────────────────────────────────────────────
@app.post("/extract-initial-consult-note", response_model=InitialConsultNoteResponse)
async def extract_initial_consult_note(request: TranscriptWithContext):
    context_blob = f"""Prior Notes:\n{request.prior_notes}\n\nTranscript:\n{request.transcript}"""
    
    system_prompt = """
    You are a hospital-based consultant writing an initial inpatient consult note. You are evaluating a patient admitted under another service.

    Extract the following based on transcript:

    – consult_question: What is the reason for consult? What issue is the primary team requesting your input on?
    – hpi: Patient’s relevant clinical history, context of hospitalization, and associated symptoms
    – exam: Your focused physical exam findings relevant to the consult
    – ap: Your assessment and plan — address the consult question clearly, provide guidance for the primary team

    Be direct, clinically formatted, and include service-specific recommendations (e.g. nephrology, ID, cardiology). Do not include extra commentary.
    """

    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context_blob}
        ],
        tools=[{"type":"function","function": INITIAL_CONSULT_FN}],
        tool_choice={"type":"function","function":{"name": "extract_initial_consult_note"}},
        temperature=0.0
    )

    tool_calls = resp.choices[0].message.tool_calls or []
    args = json.loads(tool_calls[0].function.arguments)
    return InitialConsultNoteResponse(**args)



# ─── Follow-Up Consult Extraction ──────────────────────────────────────────
@app.post("/extract-consult-followup-note", response_model=FollowUpConsultNoteResponse)
async def extract_consult_followup_note(request: TranscriptWithContext):
    context_blob = f"""Prior Notes:\n{request.prior_notes}\n\nTranscript:\n{request.transcript}"""

    system_prompt = """
    You are a hospital-based consultant writing a follow-up consult note. You previously evaluated this patient during their hospitalization, and this is a re-assessment.

    Your goal is to update your note based on new findings, treatment response, and clinical evolution. Extract the following:

    – consult_question: Persistent consult question from the initial request
    – interval_history: Changes in the patient’s condition since your prior evaluation
    – exam: New physical exam findings relevant to the consult
    – assessment: Clinical interpretation and diagnostic reasoning
    – plan: Updated guidance or recommendations for the primary team
    – follow_up: Any additional actions, monitoring, or revisit triggers

    Stay concise and clinically formatted. Treat this note as an update to a prior consult, not a new consult.
    """

    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context_blob}
        ],
        tools=[{"type":"function","function": FOLLOWUP_CONSULT_FN}],
        tool_choice={"type":"function","function":{"name": "extract_consult_followup_note"}},
        temperature=0.0
    )

    tool_calls = resp.choices[0].message.tool_calls or []
    args = json.loads(tool_calls[0].function.arguments)
    return FollowUpConsultNoteResponse(**args)



# main.py — Part 5 of 5: A&P Generators, Summaries, Exam Formatter, Drug Search
# ─── Inpatient Daily Progress A&P ───────────────────────────────────────────
@app.post("/generate-progress-ap", response_model=SummaryResponse)
async def generate_progress_ap(request: ContextualAPRequest):
    system_prompt = """
You are an expert hospitalist generating a Daily Assessment & Plan for an inpatient.
Review the PAST CLINICAL NOTES for context, then use the CURRENT DAY’S subjective and objective information
to create a concise, problem-based Assessment & Plan for the next 24 hours.

## Past Clinical Notes:
{past}

## Current Progress Note Information:
{current}
""".strip().format(
        past=request.past_notes_context,
        current=request.current_note_context
    )

    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": ""}
        ],
        temperature=0.3
    )
    return SummaryResponse(summary=resp.choices[0].message.content.strip())


# ─── Clinic Visit A&P ────────────────────────────────────────────────────────
@app.post("/generate-clinic-ap", response_model=SummaryResponse)
async def generate_clinic_ap(request: ContextualAPRequest):
    system_prompt = """
You are an expert outpatient clinician generating a concise, problem-based Assessment & Plan.
Review the PAST NOTES for context, then use the CURRENT VISIT INFORMATION to draft clear next steps, recommendations,
and any necessary screenings.

## Past Notes:
{past}

## Current Visit Information:
{current}
""".strip().format(
        past=request.past_notes_context,
        current=request.current_note_context
    )

    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": ""}
        ],
        temperature=0.3
    )
    return SummaryResponse(summary=resp.choices[0].message.content.strip())


# ─── Consult Team A&P ────────────────────────────────────────────────────────
@app.post("/generate-consult-ap", response_model=SummaryResponse)
async def generate_consult_ap(request: ContextualAPRequest):
    system_prompt = """
You are an expert hospitalist consultant generating a succinct, problem-based Assessment & Plan.
Review the PAST CONSULT NOTES for context, then use the CURRENT CONSULT INFORMATION to address the consult question directly.

## Past Consult Notes:
{past}

## Current Consult Information:
{current}
""".strip().format(
        past=request.past_notes_context,
        current=request.current_note_context
    )

    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": ""}
        ],
        temperature=0.3
    )
    return SummaryResponse(summary=resp.choices[0].message.content.strip())


# ─── Follow-Up Consult: plain transcript → structured fields ──────────────────
@app.post("/process-follow-up-consult-from-text", response_model=FollowUpConsultNoteResponse)
async def process_follow_up_consult_from_text(request: TextRequest):
    raw = (request.text or "").strip()
    if not raw:
        raise HTTPException(400, detail="Provide 'text'")

    tagged = tag_patient_speech(raw)

    system_prompt = """
You are a hospital-based consultant writing a FOLLOW-UP consult note.
Extract ONLY the requested fields using the provided tool. Be terse, clinically specific,
and focus on changes since the last consult. If information is not present, return an empty string.

Definitions:
- consult_question: restate/retain the active consult question if apparent
- interval_history: what changed since prior consult (symptoms, events, response to therapy)
- exam: focused relevant findings today
- assessment: diagnostic interpretation in the context of consult
- plan: concrete, directive steps (orders, drug/dose/route/frequency, monitoring, contingencies)
- follow_up: bullet follow-ups and when to re-evaluate or sign off

Return via tool call only.
""".strip()

    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": tagged}
        ],
        tools=[{"type":"function","function": FOLLOWUP_CONSULT_FN}],
        tool_choice={"type":"function","function":{"name":"extract_consult_followup_note"}},
        temperature=0.0
    )

    calls = resp.choices[0].message.tool_calls or []
    if not calls:
        raise HTTPException(500, detail="Follow-up consult extraction failed (no tool call).")

    args = json.loads(calls[0].function.arguments)
    return FollowUpConsultNoteResponse(**args)


# ─── Interval History (consult follow-up & progress) ──────────────────────────
@app.post("/generate-interval-history", response_model=SummaryResponse)
async def generate_interval_history(request: IntervalHistoryRequest):
    prior_blob = ""
    if request.prior_notes:
        try:
            # small, readable context; keep it compact to avoid prompt bloat
            snips = []
            for n in request.prior_notes[-5:]:
                ts = n.get("timestamp", "")
                t  = n.get("type", "")
                ap = n.get("ap", "") or n.get("assessment_plan", "") or ""
                h  = n.get("hpi", "") or n.get("subjective", "") or ""
                delta = n.get("intervalHistory", "") or n.get("interval_history", "") or ""
                piece = f"NOTE: {ts} — {t}\nHPI/Subjective: {h[:500]}\nInterval: {delta[:500]}\nA&P: {ap[:800]}"
                snips.append(piece)
            prior_blob = "\n\n---\n\n".join(snips)
        except Exception:
            prior_blob = str(request.prior_notes)[:4000]

    mode = request.mode.lower().strip()
    if mode not in ("consult_followup", "progress"):
        mode = "consult_followup"

    role_line = (
        "You are the consulting service updating your follow-up note."
        if mode == "consult_followup"
        else "You are the inpatient team writing today’s progress note."
    )

    consult_hint = f"\nActive consult question: {request.consult_question}\n" if request.consult_question else ""

    sys_prompt = f"""
{role_line}

Write a concise INTERVAL HISTORY that focuses on changes since the last note, response to therapy,
new or resolved symptoms, objective changes (vitals/labs/imaging), and actionable details.
Be concrete and clinically specific; avoid generic fluff.

Format:
- A short paragraph (2–5 sentences) OR tight bullets if clearer.
- Prioritize items relevant to the active problems{ ' and the consult question' if request.consult_question else '' }.
- If nothing changed, say so briefly and state what was actively checked.

Return via tool call; do not include extra commentary.
""".strip()

    tagged = tag_patient_speech(request.transcript)

    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": f"{consult_hint}\n\nPRIOR CONTEXT (last ~5 notes):\n{prior_blob}\n\nTRANSCRIPT:\n{tagged}"}
        ],
        tools=[{"type":"function","function": INTERVAL_HISTORY_FN}],
        tool_choice={"type":"function","function":{"name":"extract_interval_history"}},
        temperature=0.0
    )

    calls = resp.choices[0].message.tool_calls or []
    if not calls:
        raise HTTPException(500, detail="Interval history extraction failed")

    args = json.loads(calls[0].function.arguments)
    # We return only the main field as SummaryResponse to fit your iOS usage pattern,
    # but you can expose the other fields if you want later.
    return SummaryResponse(summary=args.get("interval_history", "").strip())


# ─── Follow-Up Recommendations (EBM bullets) ─────────────────────────────────
class FollowUpRecsRequest(BaseModel):
    current_note_context: str
    past_notes_context: Optional[str] = ""
    consult_question: Optional[str] = None
    options: APOptions = APOptions()

@app.post("/generate-follow-up-recommendations", response_model=SummaryResponse)
async def generate_follow_up_recommendations(request: FollowUpRecsRequest):
    cq   = f"\nConsult Question: {request.consult_question}" if request.consult_question else ""
    opts = request.options

    # Format + policy: grouped by responsible party, tight bullets, no duplication
    sys = f"""
You are an evidence-based consultant generating FOLLOW-UP RECOMMENDATIONS.

OUTPUT FORMAT (grouped; no header text):
<Responsible party>:
• <action> — <timing/window> <(why/risk) optional>
• ...
<Responsible party 2>:
• ...
Patient:
• ...
Contingencies:
• ...

REQUIREMENTS
- **Group by responsible party**, not by task. Use at most **4 groups** total (e.g., Primary team/Hospitalist, Consulting service, Patient, Contingencies). Avoid creating many specialty silos—**consolidate** when appropriate.
- Within each group, use **1–3 short bullets**. Keep total ≤ **10 bullets**.
- **No duplicate roles** across groups. If two bullets go to the same party, keep them in the same group.
- **Actionable** and **expect-level**: include responsible party + timing + what success looks like.
- **Prioritize safety**: monitoring targets and clear **escalation thresholds** (what to watch and when to call/escalate).
- { 'Include doses and monitoring targets when recommending meds.' if (opts.include_dosing or opts.include_monitoring) else '' }
- { 'You may add brief guideline hints in parentheses, e.g., (IDSA 2020); never include URLs.' if opts.cite_guideline_hints else '' }
- **Commit**; avoid vague “consider/as needed”. If truly conditional, state the condition.
- Only propose additional specialists if clearly indicated by the context; otherwise keep follow-up under the primary/consulting service.

STYLE
- Be concise and clinical. Use the exact bullet and group format above.
- Do **not** repeat the same instruction in different groups.
"""

    user = f"""
PAST CONTEXT (optional, brief):
{(request.past_notes_context or '')[:6000]}

CURRENT CONTEXT (authoritative):
{request.current_note_context[:8000]}
{cq}
""".strip()

    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": sys},
            {"role": "user",   "content": user}
        ],
        temperature=0.1
    )

    # Normalize tiny spacing things the model might emit
    out = (resp.choices[0].message.content or "").strip()
    # Ensure single blank line between groups; strip trailing whitespace
    lines = [ln.rstrip() for ln in out.splitlines()]
    normalized = []
    last_was_group_header = False
    for ln in lines:
        if ln.endswith(":") and "•" not in ln:
            if normalized and normalized[-1] != "": normalized.append("")  # blank line before a new group
            normalized.append(ln)
            last_was_group_header = True
        else:
            normalized.append(ln)
            last_was_group_header = False
    final = "\n".join(normalized).strip()

    return SummaryResponse(summary=final if final else out)


# ─── ED Brief HPI and ROS ───────────────────────────────────────────
@app.post("/extract-ed-history-from-text", response_model=EDHistoryResponse)
async def extract_ed_history_from_text(request: TextRequest):
    tagged = tag_patient_speech(request.text)

    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an ED scribe. Extract only HPI and ROS."},
            {"role": "user",   "content": tagged}
        ],
        tools=[{"type": "function", "function": EXTRACT_ED_HISTORY_FN}],
        tool_choice={"type": "function", "function": {"name": "extract_ed_history"}},
        temperature=0.0
    )

    calls = resp.choices[0].message.tool_calls or []
    if not calls:
        raise HTTPException(status_code=500, detail="History extraction failed")

    args = json.loads(calls[0].function.arguments)
    return EDHistoryResponse(**args)


# ─── ED MDM Summary ───────────────────────────────────────────
@app.post("/generate-ed-summary-from-text", response_model=EDSummaryResponse)
async def generate_ed_summary_from_text(request: TextRequest):
    """
    Hybrid ED endpoint:
    - A&P: generated via AP-v2 (context="ed") for consistent, directive plans.
    - MDM / Disposition / Billing: extracted via tool call tuned for ED.
    """
    transcript_raw = (request.text or "").strip()
    if not transcript_raw:
        raise HTTPException(status_code=400, detail="Provide 'text'")

    # Light safety: cap very long inputs and tag speakers to reduce hallucinations
    transcript = transcript_raw[:8000]
    tagged = tag_patient_speech(transcript)

    # ── 1) Generate A&P using AP-v2 (context="ed") ────────────────────────────
    ap_text = ""
    try:
        ap_req = GenerateAPV2Request(
            context="ed",
            current_note_context=f"ED Transcript:\n{tagged}",
            past_notes_context="",
            consult_question=None,
            patient_factors=None,
            options=APOptions(
                insist_specificity=True,
                include_dosing=True,
                include_monitoring=True,
                include_disposition=True,
                include_consult_triggers=True,
                cite_guideline_hints=False,
                avoid_handwaving=True,
                force_commitment=True
            )
        )
        ap_v2 = await generate_ap_v2(ap_req)  # reuse in-process route function
        ap_text = (ap_v2.summary or "").strip()
    except Exception as e:
        log_event("WARNING", "ed_ap_v2_failed", error=str(e))
        ap_text = ""  # fall back to tool-call A&P if needed

    # ── 2) Extract MDM / Dispo / Billing via tool call (with upgraded prompt) ─
    system_prompt = """
You are an emergency physician documenting a patient encounter in the ED.

Return results using the `extract_ed_summary` tool with these expectations:

- assessment_and_plan:
  - Problem-based format: `# Problem` with short, **directive** bullets.
  - For each active problem, specify exact **orders** and **therapies**:
    - **Medications**: name, dose, route, frequency, typical duration (e.g., ceftriaxone 1 g IV once, then doxycycline 100 mg PO BID × 7d).
    - **Initial orders**: labs, imaging, EKG, cultures with a one-line rationale (e.g., “lactate to trend sepsis severity”).
    - **Decision rules / risk scores** when relevant (HEART for chest pain, Wells/YEARS/PERC for PE, CURB-65 for PNA, GBS for UGIB, Ottawa, NEXUS/CCR, etc.).
    - **Monitoring & reassessment** (what to re-check and when).
    - **Consult / escalation triggers** (service + threshold values).
    - Provide first-line and at least one reasonable **alternative** for penicillin allergy, pregnancy, and eGFR < 30 when applicable.
  - Default to **ED initial dosing** (first dose now). If longer-course decisions are outside ED scope, say so explicitly.
  - If a needed detail is missing, **state the assumption** (e.g., “no allergy history provided”) and still commit.

- mdm:
  - Differential with **most-likely** vs **can’t-miss** items and brief justification.
  - Tie diagnostics and therapies to the differential and risk-score outputs.
  - Document data reviewed (prior notes, EMS, vitals trends), complexity, and risks that justify the billing level.
  - Do **not** fabricate results; if tests are proposed but not resulted, list under orders, not results.

- disposition_recommendation:
  - Admit / observe / discharge with **explicit criteria** that push one way or another.
  - Include **return precautions** and **follow-up timing**.

- billing_level_suggestion:
  - Suggest an E/M level (e.g., 99284) with a one-sentence justification based on problems/risks/data.

Global rules:
- Use only information present in the transcript. If something is missing, say the assumption aloud and proceed.
- Be specific and directive; avoid vague “consider / as needed” unless truly conditional, and define the condition.
"""

    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": transcript},
        ],
        tools=[{"type": "function", "function": EXTRACT_ED_SUMMARY_FN}],
        tool_choice={"type": "function", "function": {"name": "extract_ed_summary"}},
        temperature=0.1
    )

    tool_calls = resp.choices[0].message.tool_calls or []
    if not tool_calls:
        log_event("ERROR", "ed_summary_extraction_failed", detail="AI response contained no tool calls.")
        raise HTTPException(status_code=500, detail="ED summary extraction failed to produce a tool call.")

    try:
        args = json.loads(tool_calls[0].function.arguments)
    except (json.JSONDecodeError, IndexError) as e:
        raw_args = tool_calls[0].function.arguments if tool_calls else "N/A"
        log_event("ERROR", "ed_summary_parsing_failed", error=str(e), raw_args=raw_args)
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response for ED summary: {e}")

    # Prefer AP-v2 output if available; otherwise fall back to tool’s A&P
    final_ap = ap_text or args.get("assessment_and_plan", "")

    return EDSummaryResponse(
        assessment_and_plan=final_ap,
        mdm=args.get("mdm", ""),
        disposition_recommendation=args.get("disposition_recommendation", ""),
        billing_level_suggestion=args.get("billing_level_suggestion", "")
    )


# ─── Generic Summary ────────────────────────────────────────────────────────
@app.post("/generate-summary", response_model=SummaryResponse)
async def generate_summary(request: PromptRequest):
    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[{"role": "user", "content": request.prompt}],
        temperature=0.4
    )
    return SummaryResponse(summary=resp.choices[0].message.content.strip())


# ─── Hospital Course Summary ─────────────────────────────────────────────────
@app.post("/generate-hospital-summary", response_model=SummaryResponse)
async def generate_hospital_summary(notes: List[Note]):
    if not notes:
        raise HTTPException(status_code=400, detail="Need at least one note")

    formatted = "\n\n".join(
        f"🗓 {n.timestamp.strftime('%Y-%m-%d %H:%M')} — {n.type}\n"
        f"HPI: {n.hpi or n.subjective or 'N/A'}\n"
        f"Exam: {n.exam or 'N/A'}\n"
        f"Objective: {n.objective or 'N/A'}"
        for n in notes
    )

    system_prompt = """
You are a hospitalist summarizing a patient's hospital course.
Please generate:
1. A 2–4 sentence overview of the hospital stay.
2. A bulleted list of major clinical events.
3. A short list of pending items or follow‐up needs.
""".strip()

    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": formatted}
        ],
        temperature=0.4
    )
    return SummaryResponse(summary=resp.choices[0].message.content.strip())


# ─── Discharge Summary ───────────────────────────────────────────────────────
@app.post("/generate-discharge-summary", response_model=SummaryResponse)
async def generate_discharge_summary(notes: List[SummaryNote]):
    if not notes:
        raise HTTPException(status_code=400, detail="Need at least one note")

    formatted = "\n\n".join(
        f"Date: {n.timestamp.strftime('%Y-%m-%d')}\n"
        f"HPI: {n.hpi or 'N/A'}\n"
        f"Exam: {n.exam or 'N/A'}\n"
        f"A&P: {n.ap or 'N/A'}\n"
        f"MDM: {n.mdm or 'N/A'}"
        for n in notes
    )

    system_prompt = """
You are a discharge planning assistant. Based on the following clinical notes, generate a concise, structured discharge summary.
Include:
- Hospital course
- Treatments and interventions
- Clinical progression
- Discharge instructions and follow‐up plan
""".strip()

    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": formatted}
        ],
        temperature=0.5
    )
    return SummaryResponse(summary=resp.choices[0].message.content.strip())


# ─── MDM Generator ─────────────────────────────────────────────────────────
@app.post("/generate-mdm", response_model=MDMResponse)
async def generate_mdm(notes: List[Note]):
    note = notes[-1]
    clinical_context = (
        f"Patient Summary:\n"
        f"- Timestamp: {note.timestamp}\n"
        f"- HPI: {note.hpi or note.subjective or 'N/A'}\n"
        f"- Exam: {note.exam or 'N/A'}\n"
        f"- Labs/Imaging: {note.objective or 'N/A'}\n"
        f"- Assessment/Plan: {note.ap or 'N/A'}\n\n"
        "Generate a clear, medically sound Medical Decision Making (MDM) paragraph based on this clinical context."
    )

    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[{"role": "user", "content": clinical_context}],
        temperature=0.3
    )
    return MDMResponse(mdm=resp.choices[0].message.content.strip())


# ─── Enhance A&P ───────────────────────────────────────────────────────────
@app.post("/enhance-ap", response_model=SummaryResponse)
async def enhance_ap(request: TextRequest):
    system_prompt = """
You are an experienced hospitalist. Rewrite the following Assessment & Plan into exactly this structure:
Assessment: <one-sentence summary>
# <Problem 1>
• <first bullet plan item>
• <second bullet plan item>
…
# <Problem 2>
…
Do not invent new problems—use only those present below.

Original A&P:
""".strip()

    content = system_prompt + "\n" + request.text
    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        temperature=0.4
    )
    return SummaryResponse(summary=resp.choices[0].message.content.strip())


# ─── Unified, EBM-leaning A/P generator (v2) ─────────────────────────────────
@app.post("/generate-ap-v2", response_model=SummaryResponse)
async def generate_ap_v2(request: GenerateAPV2Request):
    o = request.options
    knobs = []
    if o.insist_specificity:    knobs.append("Be specific and directive (avoid vagueness).")
    if o.include_dosing:        knobs.append("Include medication name, dose, route, frequency, and typical duration when appropriate.")
    if o.include_monitoring:    knobs.append("Include monitoring targets and clear stop/step-down criteria.")
    if o.include_disposition:   knobs.append("State disposition level (home/observation/inpatient) with rationale when applicable.")
    if o.include_consult_triggers: knobs.append("Add explicit consult/escalation triggers (who to call, based on which thresholds).")
    if o.cite_guideline_hints:  knobs.append("Add brief parenthetical guideline hints (e.g., 'IDSA 2020', 'AHA 2021'); do NOT add URLs.")
    if o.avoid_handwaving:      knobs.append("Avoid 'consider/if needed' phrasing; commit to a plan and justify briefly.")
    if o.force_commitment:      knobs.append("When multiple reasonable options exist, choose one and state why.")

    context_line = f"Context: {request.context.upper()}"
    cq = f"\nConsult Question: {request.consult_question}" if request.consult_question else ""
    pf = f"\nPatient Factors: {request.patient_factors}" if request.patient_factors else ""

    sys = f"""
You are an expert clinician generating a **problem-based Assessment & Plan** as a numbered list.
{context_line}
Rules:
- Start with an **Assessment** one-liner (who/what/why-now).
- Then list problems as **# Problem:** followed by concrete plan bullets.
- Keep each bullet short but specific. Use common inpatient/outpatient defaults safely.
- Include rationale when it drives the choice (e.g., CURB-65 for PNA, QT risk, renal dosing).
- If information is missing, state assumptions explicitly and proceed.

Tuning:
- {' '.join(knobs)}
""".strip()

    user = f"""
PAST NOTES (optional):
{(request.past_notes_context or '')[:6000]}

CURRENT NOTE:
{request.current_note_context[:8000]}
{cq}{pf}
""".strip()

    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": sys},
            {"role": "user",   "content": user}
        ],
        temperature=0.2
    )

    return SummaryResponse(summary=(resp.choices[0].message.content or "").strip())


# ─── Exam Formatter ─────────────────────────────────────────────────────────
@app.post("/format-exam", response_model=SummaryResponse)
async def format_exam(request: TextRequest):
    system_prompt = """
You are an expert medical scribe. Convert the following dictated physical exam into a structured note.
Use standard headings: Constitutional, HEENT, Cardiovascular, Respiratory, Abdominal, Extremities, Neurological, Skin.
Do not add information that wasn't dictated.

Dictation:
""".strip()

    content = system_prompt + "\n\"\"\"\n" + request.text + "\n\"\"\""
    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        temperature=0.2
    )
    return SummaryResponse(summary=resp.choices[0].message.content.strip())


# ─── Dictation Formatter ─────────────────────────────────────────────────────────
@app.post("/enhance-text-with-dictation", response_model=EnhanceTextResponse)
async def enhance_text_with_dictation(req: EnhanceTextRequest):
    system = """
You merge new dictation into an existing clinical paragraph.
- Preserve existing structure and clinical facts.
- Insert new information in the right place, revise conflicting facts, and remove redundant lines.
- Keep the output concise and cleanly formatted; plain text only.
"""
    user = f"ORIGINAL:\n{req.original_text}\n\nNEW DICTATION:\n{req.new_dictation}"
    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.1
    )
    return EnhanceTextResponse(summary=(resp.choices[0].message.content or "").strip())


# ─── Merge Note Universal ────────────────────────────────────────────────────
@app.post("/merge-note-from-text")
async def merge_note_from_text(request: MergeNoteRequest):
    """
    Merge existing structured note + new dictation into a unified updated note.
    Supports admission, progress, outpatient, consult, and ED.
    For progress/consult, incorporates past notes for continuity.
    """

    # Merge rules
    system_prompt = f"""
    You are a medical scribe updating a {request.note_type} note.

    MERGE RULES:
    - Start from the EXISTING NOTE JSON (all fields may be present).
    - Integrate NEW TEXT into the correct fields.
    - Preserve original info unless the new text explicitly updates/replaces it.
    - If new text provides updated info (e.g., revised HPI, meds), replace that field.
    - If new text provides missing info, fill it in.
    - For progress/consult, consider continuity context from past notes if available.
    - Always return a full JSON object with ALL fields for this note type.
    - Never output free text outside JSON.
    """

    # Past notes (only for progress/consult continuity)
    past_blob = ""
    if request.note_type in ("progress", "consult") and request.past_notes:
        try:
            snippets = []
            for n in request.past_notes[-5:]:
                ts = n.get("timestamp", "")
                t  = n.get("type", "")
                ap = n.get("ap", "") or n.get("assessment_plan", "") or ""
                h  = n.get("hpi", "") or n.get("subjective", "") or ""
                delta = n.get("intervalHistory", "") or n.get("interval_history", "") or ""
                piece = f"NOTE: {ts} — {t}\nHPI/Subjective: {h[:500]}\nInterval: {delta[:500]}\nA&P: {ap[:800]}"
                snippets.append(piece)
            past_blob = "\n\n---\n\n".join(snippets)
        except Exception:
            past_blob = str(request.past_notes)[:4000]

    # Messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"EXISTING NOTE:\n{json.dumps(request.existing_note)}"},
        {"role": "user", "content": f"NEW TEXT:\n{request.new_text}"}
    ]
    if past_blob:
        messages.append({"role": "user", "content": f"PAST NOTES CONTEXT:\n{past_blob}"})

    # Tool mapping
    tool_map = {
        "admission": (EXTRACT_HNP_FN, "extract_hnp"),
        "progress": (PROGRESS_NOTE_FN, "extract_progress_note"),
        "outpatient": (EXTRACT_INITIAL_CLINIC_FN, "extract_initial_clinic_note"),
        "consult": (INITIAL_CONSULT_FN, "extract_initial_consult_note"),
        "ed": (EXTRACT_ED_HISTORY_FN, "extract_ed_history")
    }
    if request.note_type not in tool_map:
        raise HTTPException(400, detail=f"Unsupported note_type: {request.note_type}")

    tool_fn, tool_choice_name = tool_map[request.note_type]

    # AI call
    resp = await ai_completion_or_fail(
        model="gpt-4o",
        messages=messages,
        tools=[{"type": "function", "function": tool_fn}],
        tool_choice={"type": "function", "function": {"name": tool_choice_name}},
        temperature=0.0
    )

    tool_calls = resp.choices[0].message.tool_calls or []
    if not tool_calls:
        raise HTTPException(500, detail="Merge failed: no tool calls.")

    args = json.loads(tool_calls[0].function.arguments)
    return args


# ─── Drug Search (RxNav) ────────────────────────────────────────────────────
def extract_problem_headings_py(text: str) -> List[str]:
    headings = []
    for line in text.splitlines():
        clean_line = line.strip()
        if not clean_line:
            continue
        
        # Markdown-style: # Problem
        if clean_line.startswith("#"):
            headings.append(re.sub(r"^\s*#+\s*", "", clean_line))
            continue
            
        # Numbered list: 1) Problem or 1. Problem
        if re.match(r"^\s*\d+[\.\)]\s+", clean_line):
            headings.append(re.sub(r"^\s*\d+[\.\)]\s+", "", clean_line))
            continue
            
        # Bulleted list: • Problem or - Problem
        if clean_line.startswith("•") or clean_line.startswith("-"):
            headings.append(clean_line[1:].strip())
            continue
            
    # Deduplicate while preserving order
    return list(dict.fromkeys(headings))

@app.post("/generate-updated-plan", response_model=SummaryResponse)
async def generate_updated_plan(request: UpdatedPlanRequest):
    new_ap = request.new_ap.strip()
    prior_ap_lower = (request.prior_ap or "").lower()
    
    if not new_ap:
        return SummaryResponse(summary="No changes from prior plan.")
        
    problems = extract_problem_headings_py(new_ap)
    
    if problems:
        lines = ["Plan changes since last note:"]
        for p in problems:
            tag = "" if not prior_ap_lower else " (updated)" if p.lower() in prior_ap_lower else " (new)"
            lines.append(f"• {p}{tag}")
        return SummaryResponse(summary="\n".join(lines))
        
    # Fallback if no headings are found
    bullets = [f"• {line.strip()}" for line in new_ap.splitlines() if line.strip()][:12]
    summary = "Plan updates:\n" + "\n".join(bullets) if bullets else "No changes from prior plan."
    return SummaryResponse(summary=summary)


# ─── Drug Search (RxNav) ────────────────────────────────────────────────────
@app.get("/search-drugs", response_model=DrugSearchResponse)
async def search_drugs(query: str):
    if not query.strip():
        return DrugSearchResponse(results=[])

    q = quote(query)
    url = f"https://rxnav.nlm.nih.gov/REST/drugs.json?name={q}"
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()

    results = []
    for group in data.get("drugGroup", {}).get("conceptGroup", []):
        for concept in group.get("conceptProperties", []):
            results.append({
                "id":           concept.get("rxcui", ""),
                "name":         concept.get("name", ""),
                "isControlled": False
            })

    return DrugSearchResponse(results=results)
