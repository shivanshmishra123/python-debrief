from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from schema import MeetingPayload, QueryPayload
from text_processor import get_overlapping_chunks
from google import genai
from google.genai import types
from pinecone import Pinecone
import os
from dotenv import load_dotenv 

load_dotenv() 

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

class ExtractionResponse(BaseModel):
    items: List[ExtractedItemSchema]

class TranscriptRequest(BaseModel):
    transcript: str

# ==========================================
# UNIFIED STRUCTURED EXTRACTION ENDPOINT
# ==========================================
@app.post("/api/extract-structured", response_model=ExtractionResponse)
async def extract_structured_data(payload: TranscriptRequest):
    print("\n🧩 Extracting structured data from transcript using Gemini...")
    
    # Blended prompt utilizing the robust corporate analyst guidelines
    system_instruction = """
    You are an expert corporate meeting analyst. Your task is to process meeting transcripts and extract key structured data.

    Analyze the provided transcript and extract items into exactly four categories:
    1. DECISION: Finalized choices, approvals, or strategic directions agreed upon.
    2. ACTION_ITEM: Specific tasks assigned to individuals or teams.
    3. OPEN_QUESTION: Unresolved issues, blockers, or topics tabled for future discussion.
    4. KEY_CONTEXT: Crucial rationales, important updates, or underlying reasons for decisions that do not fit the above categories but are vital for historical memory.

    RULES:
    - Do not summarize the entire meeting.
    - Extract atomic, self-contained points.
    """
    
    try:
        response = gemini_client.models.generate_content(
            model='gemini-3.5-flash',
            contents=payload.transcript,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=ExtractionResponse,
                temperature=0.2  # Kept low for high analytical accuracy
            )
        )
        print("✅ Structured extraction complete!")
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
        model="gemini-3.5-flash",
        contents=prompt
    )

    
    print("✅ Answer generated successfully!")

    return {
        "question": payload.question,
        "answer": llm_response.text,
        "sources_used": len(retrieved_chunks)
    }