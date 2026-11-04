import os
import json
from typing import Dict, Any, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app import TicketSimilarityEngine

app = FastAPI(title="Solari Support Similarity API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TicketQuery(BaseModel):
    subject: str
    description: str
    category: Optional[str] = ""

engine = TicketSimilarityEngine(confidence_threshold=0.20)

@app.on_event("startup")
def startup_event():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    historical_path = os.path.join(base_dir, "starter_kit", "data", "tickets.json")
    with open(historical_path, "r", encoding="utf-8") as f:
        historical_tickets = json.load(f)["tickets"]
    engine.fit(historical_tickets)

# Serve production index.html directly from root URL
@app.get("/", response_class=FileResponse)
def read_root():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "index.html")

@app.post("/api/match")
def match_ticket(query: TicketQuery) -> Dict[str, Any]:
    payload = query.dict()
    matches = engine.find_similar(payload, top_n=3)
    return {
        "query": payload,
        "has_match": len(matches) > 0,
        "total_matches": len(matches),
        "matches": matches
    }