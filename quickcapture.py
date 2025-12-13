# quickcapture.py
import os, io, json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from firebase_admin import storage
from openai import AsyncOpenAI

router = APIRouter()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
bucket = storage.bucket()


class QuickCaptureRequest(BaseModel):
    file_path: str
    note_type: str


class QuickCaptureResponse(BaseModel):
    transcript: str
    note: dict


ALLOWED_NOTE_TYPES = {
    "admission", "progress", "consult",
    "outpatient", "ed", "bedside"
}


@router.post("/v1/quickcapture-audio", response_model=QuickCaptureResponse)
async def quickcapture_audio(req: QuickCaptureRequest):

    if req.note_type not in ALLOWED_NOTE_TYPES:
        raise HTTPException(400, f"Invalid note_type: {req.note_type}")

    # 1️⃣ Download audio
    try:
        audio_bytes = bucket.blob(req.file_path).download_as_bytes()
    except Exception as e:
        raise HTTPException(400, f"Failed to load audio: {e}")

    audio_buf = io.BytesIO(audio_bytes)
    audio_buf.name = "audio.m4a"

    # 2️⃣ Transcribe
    try:
        tr = await client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_buf
        )
        transcript = tr.text.strip()
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")

    if len(transcript.split()) < 40:
        raise HTTPException(400, "Transcript too short for QuickCapture")

    # 3️⃣ Compress
    compress_prompt = f"""
Summarize the following clinician–patient conversation into
a concise, clinically relevant narrative.

Rules:
- Remove repetition and filler
- Preserve symptoms, timelines, PMH, meds, ROS, exam mentions
- Do NOT invent facts
- Output plain text only

TRANSCRIPT:
{transcript}
"""

    compressed = (
        await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=1200,
            messages=[{"role": "system", "content": compress_prompt}],
        )
    ).choices[0].message.content.strip()

    # 4️⃣ Structure note
    structure_prompt = f"""
You are a senior clinician generating a {req.note_type} note.

Generate a complete structured medical note from the context below.

Rules:
- Do NOT hallucinate
- Professional clinical language
- Return ONLY valid JSON

CONTEXT:
{compressed}
"""

    structured = await client.chat.completions.create(
        model="gpt-4o",
        temperature=0.1,
        max_tokens=2000,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": structure_prompt}],
    )

    return QuickCaptureResponse(
        transcript=transcript,
        note=json.loads(structured.choices[0].message.content)
    )
