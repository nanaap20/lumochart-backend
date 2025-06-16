from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# ✅ Add root endpoint
@app.get("/")
async def root():
    return {"message": "LumoChart backend is running 🎉"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class PromptRequest(BaseModel):
    prompt: str

class Note(BaseModel):
    timestamp: str
    hpi: Optional[str] = None
    exam: Optional[str] = None
    ap: Optional[str] = None
    mdm: Optional[str] = None
    type: Optional[str] = None
    ros: Optional[str] = None
    subjective: Optional[str] = None
    objective: Optional[str] = None
    summary: Optional[str] = None
    ownerId: Optional[str] = None
    patientID: Optional[str] = None

# Endpoint: general-purpose summary
@app.post("/generate-summary")
async def generate_summary(request: PromptRequest):
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[{"role": "user", "content": request.prompt}],
            temperature=0.4
        )
        return {"summary": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}

# Endpoint: discharge summary
@app.post("/generate-discharge-summary")
async def generate_discharge_summary(notes: List[Note]):
    try:
        print("🧾 Received notes:")
        for n in notes:
            print(f"Date: {n.timestamp} | HPI: {n.hpi}")

        formatted_notes = "\n\n".join([
            f"Date: {n.timestamp}\n"
            f"HPI: {n.hpi or 'N/A'}\n"
            f"Exam: {n.exam or 'N/A'}\n"
            f"A&P: {n.ap or 'N/A'}\n"
            f"MDM: {n.mdm or 'N/A'}"
            for n in notes
        ])

        prompt = f"""
You are a discharge planning assistant. Based on the following clinical notes, generate a concise, structured discharge summary. Include:
- Hospital course
- Treatments and interventions
- Clinical progression
- Discharge instructions and follow-up plan

Notes:
{formatted_notes}
"""

        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return {"summary": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}

# Endpoint: hospital course summary
@app.post("/generate-hospital-summary")
async def generate_hospital_summary(notes: List[Note]):
    try:
        print("📥 Generating hospital summary from notes:", len(notes))

        formatted_notes = "\n\n".join([
            f"🗓 {n.timestamp} — {n.type or 'Unknown Type'}\n"
            f"HPI: {n.hpi or n.subjective or 'N/A'}\n"
            f"Exam: {n.exam or 'N/A'}\n"
            f"Plan: {n.ap or 'N/A'}"
            for n in notes
        ])

        prompt = f"""
You are a hospitalist summarizing a patient's hospital course. Please generate:
1. A 2-4 sentence overview of the hospital stay.
2. A bulleted list of major clinical events (diagnoses, treatments, status changes).
3. A short list of pending items or follow-up needs.

Here are the progress notes:

{formatted_notes}
"""

        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return {"summary": response.choices[0].message.content}
    except Exception as e:
        print("❌ Hospital summary generation failed:", e)
        return {"error": str(e)}
# Endpoint: generate MDM
@app.post("/generate-mdm")
async def generate_mdm(note: Note):
    try:
        clinical_context = f"""
Patient Summary:
- Timestamp: {note.timestamp}
- HPI: {note.hpi or note.subjective or "N/A"}
- Exam: {note.exam or "N/A"}
- Labs/Imaging: {note.objective or "N/A"}
- Assessment/Plan: {note.ap or "N/A"}

Instructions:
Generate a clear, medically sound Medical Decision Making (MDM) paragraph based on this clinical context. Include:
- Summary of clinical problem(s)
- Key findings (labs, imaging, vitals)
- Differential diagnosis (if relevant)
- Rationale for current treatment or interventions
- Clinical risk assessment
- Disposition reasoning (e.g., admit, discharge, observation)

Respond in 1 paragraph.
        """

        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[{"role": "user", "content": clinical_context}],
            temperature=0.3
        )
        return {"mdm": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}
