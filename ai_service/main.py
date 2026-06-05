from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from schema import MeetingPayload, QueryPayload
from text_processor import get_overlapping_chunks
from google import genai
from google.genai import types
from pinecone import Pinecone
import os
import tempfile

# Ensure ffmpeg bin directory is in PATH on Windows so Whisper can find it
ffmpeg_dir = r"C:\Users\13shi\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"
if os.path.exists(ffmpeg_dir) and ffmpeg_dir not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + ffmpeg_dir

import whisper
from dotenv import load_dotenv

load_dotenv()

# =============================================================
# WHISPER MODEL — loaded ONCE at startup, reused for every call
# Why? Loading the model takes ~2-5 seconds. We don't want that
# delay on every request, so we load it globally when the app starts.
# 'base' gives a good balance of speed and accuracy on CPU.
# =============================================================
print("⏳ Loading Whisper base model (first time may download ~142 MB)...")
whisper_model = whisper.load_model("base")
print("✅ Whisper model loaded and ready!")

app = FastAPI(title="Meeting Debrief AI Service")

# Initialize Clients
gemini_client = genai.Client() 
pc = Pinecone()

PINECONE_INDEX_NAME = "meeting-debrief"

# ==========================================
# STRUCTURED EXTRACTION SCHEMAS
# ==========================================
class ItemType(str, Enum):
    DECISION = "DECISION"
    ACTION_ITEM = "ACTION_ITEM"
    OPEN_QUESTION = "OPEN_QUESTION"
    KEY_CONTEXT = "KEY_CONTEXT"

class ExtractedItemSchema(BaseModel):
    type: ItemType
    content: str = Field(description="The actual extracted text for this item.")

class HistoricItem(BaseModel):
    id: str
    type: str
    content: str

class ExtractionResponse(BaseModel):
    items: List[ExtractedItemSchema]
    drifted_ids: List[str] = Field(default_factory=list, description="IDs of historic items that are now superseded, completed, or drifted.")

class TranscriptRequest(BaseModel):
    transcript: str
    history: Optional[List[HistoricItem]] = None

# ==========================================
# UNIFIED STRUCTURED EXTRACTION ENDPOINT
# ==========================================
@app.post("/api/extract-structured", response_model=ExtractionResponse)
async def extract_structured_data(payload: TranscriptRequest):
    print("\n🧩 Extracting structured data and checking for drifts...")
    
    input_contents = f"Transcript:\n{payload.transcript}"
    
    system_instruction = """
    You are an expert corporate meeting analyst. Your task is to process meeting transcripts and extract key structured data.

    Part 1: Analyze the provided transcript and extract NEW items into exactly four categories:
    1. DECISION: Finalized choices, approvals, or strategic directions agreed upon.
    2. ACTION_ITEM: Specific tasks assigned to individuals or teams.
    3. OPEN_QUESTION: Unresolved issues, blockers, or topics tabled for future discussion.
    4. KEY_CONTEXT: Crucial rationales, important updates, or underlying reasons for decisions that do not fit the above categories but are vital for historical memory.

    RULES:
    - Do not summarize the entire meeting.
    - Extract atomic, self-contained points.
    """
    
    if payload.history:
        print(f"📊 Reconciling with {len(payload.history)} historic items from thread history...")
        history_str = "\n".join([f"- ID: {item.id} | Type: {item.type} | Content: {item.content}" for item in payload.history])
        input_contents = f"Transcript:\n{payload.transcript}\n\n---\n\nHistoric Active Items:\n{history_str}"
        
        system_instruction += """
        
        Part 2: Reconcile with History:
        You are also provided with a list of "Historic Active Items" from previous meetings in this thread.
        Compare the new transcript with these historic items. Identify if any historic items have been:
        - Contradicted, updated, or superseded by a new decision or discussion (e.g. previous: "use Redis", new: "use PostgreSQL instead").
        - Completed or resolved (e.g. action item like "Dave to write migration script" is mentioned as done in the new transcript).
        
        If an item matches either condition, you MUST add its exact "ID" to the `drifted_ids` list in the response.
        Only add an ID to `drifted_ids` if it is clearly overridden or resolved by the current transcript.
        """

    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=input_contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=ExtractionResponse,
                temperature=0.2  # Kept low for high analytical accuracy
            )
        )
        print("✅ Structured extraction and drift analysis complete!")
        return ExtractionResponse.model_validate_json(response.text)
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# EXISTING: VECTOR PIPELINE (PINE CONE)
# ==========================================
@app.post("/api/extract")
async def process_meeting(payload: MeetingPayload):
    print(f"\n✅ Received Meeting from Spring Boot: {payload.title} (ID: {payload.meeting_id})")
    
    if not payload.transcript:
        raise HTTPException(status_code=400, detail="transcript is missing or empty")

    print("🔪 Chunking transcript...")
    chunks = get_overlapping_chunks(payload.transcript, chunk_size=1000, overlap=200)

    print("🧠 Generating vectors and compiling Pinecone payload...")
    vectors_to_upsert = []
    
    for i, chunk_text in enumerate(chunks):
        contextualized_text = f"Meeting Title: {payload.title}\nDate: {payload.date}\nContent: {chunk_text}"
        
        response = gemini_client.models.embed_content(
            model="gemini-embedding-2",
            contents=contextualized_text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                title=payload.title
            )
        )
        
        vector = response.embeddings[0].values
        chunk_id = f"{payload.meeting_id}-chunk-{i}"
        
        metadata = {
            "meeting_id": payload.meeting_id,
            "title": payload.title,
            "date": payload.date,
            "text": chunk_text 
        }
        
        vectors_to_upsert.append((chunk_id, vector, metadata))
        print(f"   -> Prepared chunk {i+1}/{len(chunks)}")

    print("🚀 Pushing to Pinecone...")
    index = pc.Index(PINECONE_INDEX_NAME)
    index.upsert(vectors=vectors_to_upsert)
    print("✅ Upsert complete!")

    return {
        "status": "success", 
        "message": f"Successfully vectorized and stored {len(vectors_to_upsert)} chunks for meeting {payload.meeting_id}"
    }

# ==========================================
# EXISTING: RAG QUERY PIPELINE
# ==========================================
@app.post("/api/query")
async def query_meetings(payload: QueryPayload):
    print(f"\n🔍 Received Query: '{payload.question}'")
    
    print("🧠 Embedding question...")
    response = gemini_client.models.embed_content(
        model="gemini-embedding-2",
        contents=payload.question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY" 
        )
    )
    question_vector = response.embeddings[0].values

    print("🔎 Searching Pinecone for relevant transcript chunks...")
    index = pc.Index(PINECONE_INDEX_NAME)
    
    query_params = {
        "vector": question_vector,
        "top_k": 4, 
        "include_metadata": True
    }
    
    if payload.meeting_id:
        query_params["filter"] = {"meeting_id": payload.meeting_id}

    search_results = index.query(**query_params)

    retrieved_chunks = []
    for match in search_results.matches:
        if "text" in match.metadata:
            retrieved_chunks.append(match.metadata["text"])

    if not retrieved_chunks:
        return {"answer": "I couldn't find any relevant context in the meeting transcripts to answer that question."}

    print("💬 Generating final answer with Gemini...")
    context_block = "\n\n---\n\n".join(retrieved_chunks)
    
    prompt = f"""
    You are a helpful AI Meeting Assistant. Answer the user's question using ONLY the provided meeting transcript context.
    If the answer is not contained in the context, politely say you don't know based on the transcripts. Do not guess.

    Context from Meeting(s):
    {context_block}

    User Question: {payload.question}
    """

    llm_response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    
    print("✅ Answer generated successfully!")

    return {
        "question": payload.question,
        "answer": llm_response.text,
        "sources_used": len(retrieved_chunks)
    }


# ==========================================
# AUDIO TRANSCRIPTION ENDPOINT
# ==========================================
# KEY CONCEPT: UploadFile is FastAPI's way of receiving file uploads.
# The browser sends the MP3 as multipart/form-data — FastAPI unwraps
# it and gives us the file as a Python object we can read from.
@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    print(f"\n🎙️ Received audio file: {audio.filename} ({audio.content_type})")

    # STEP 1: Validate file type
    # We only accept audio formats that Whisper supports.
    allowed_types = ["audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
                     "audio/mp4", "audio/m4a", "audio/ogg", "audio/webm"]
    if audio.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {audio.content_type}. Please upload an MP3, WAV, or M4A file."
        )

    # STEP 2: Save the uploaded bytes to a TEMPORARY file on disk
    # WHY? Whisper needs a file PATH to read from — it can't read from
    # raw bytes in memory directly. tempfile.NamedTemporaryFile creates
    # a file that automatically gets deleted when we're done.
    # delete=False means we control deletion manually (needed on Windows).
    suffix = os.path.splitext(audio.filename)[1] or ".mp3"  # preserve extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_path = tmp_file.name
        content = await audio.read()  # read all bytes from the upload
        tmp_file.write(content)       # write them to disk

    print(f"   -> Saved to temp file: {tmp_path}")

    try:
        # STEP 3: Run Whisper transcription
        # HOW WHISPER WORKS:
        # It converts audio to a mel spectrogram (a visual representation
        # of frequencies over time), then runs it through a transformer
        # model trained on 680,000 hours of audio. It outputs text.
        # fp16=False means we use 32-bit floats — required on CPU (not GPU).
        print("🧠 Running Whisper transcription...")
        result = whisper_model.transcribe(tmp_path, fp16=False)

        # result["text"] is the full raw transcript as a single string
        # result["segments"] contains word-level timestamps (for future diarization)
        transcript_text = result["text"].strip()

        print(f"✅ Transcription complete! ({len(transcript_text)} characters)")
        print(f"   Preview: {transcript_text[:200]}...")

        return {
            "transcript": transcript_text,
            "duration_seconds": result.get("segments", [{}])[-1].get("end", 0) if result.get("segments") else 0,
            "language": result.get("language", "en")
        }

    except Exception as e:
        print(f"❌ Whisper transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    finally:
        # STEP 4: Always clean up the temp file, even if an error occurred
        # The 'finally' block runs whether or not an exception was raised.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"   -> Cleaned up temp file: {tmp_path}")