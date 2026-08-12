"""
inference/api.py
================
FastAPI REST API for Resume Intelligence LLM.

Endpoints:
  GET  /health        -> Health check & system status
  POST /parse-resume  -> Accepts JSON body {"resume_text": "..."}, returns structured JSON schema

Run server:
    uvicorn inference.api:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.model_loader import parse_resume

app = FastAPI(
    title="Resume Intelligence LLM API",
    description="Domain-specific resume parsing service powered by fine-tuned Qwen3-1.7B.",
    version="1.0.0",
)


class ParseRequest(BaseModel):
    resume_text: str
    adapter_path: Optional[str] = "./adapter_v1"


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str


@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "healthy",
        "model": "Qwen/Qwen3-1.7B",
        "version": "1.0.0",
    }


@app.post("/parse-resume")
def parse_resume_endpoint(request: ParseRequest) -> Dict[str, Any]:
    if not request.resume_text or not request.resume_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="field 'resume_text' cannot be empty."
        )

    try:
        result = parse_resume(
            resume_text=request.resume_text,
            adapter_path=request.adapter_path,
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error parsing resume: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
