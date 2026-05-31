from pydantic import BaseModel
from typing import List, Optional

class Utterance(BaseModel):
    speaker: str
    text: str

class MeetingPayload(BaseModel):
    # Make the other fields optional just in case Java isn't sending them yet
    meeting_id: Optional[str] = "unknown-id"
    title: Optional[str] = "Untitled Meeting"
    date: Optional[str] = "unknown-date"
    
    # Change this from 'raw_transcript' to 'transcript' to match Java!
    transcript: str 
    
    utterances: Optional[List[Utterance]] = None

class QueryPayload(BaseModel):
    question: str
    # We make this optional so you can ask about ONE specific meeting, 
    # or leave it blank to search across ALL meetings in the database!
    meeting_id: Optional[str] = None