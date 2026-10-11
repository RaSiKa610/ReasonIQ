"""
ReasonIQ - Backend Main Entry Point

FastAPI application for the real-time AI-powered debate and argument analysis system.
Phase 1: In-memory runtime state, PyAV audio adapter, Silero VAD, faster-whisper,
ReasonIQ NLP Engine, and WebSocket broadcasting.
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend root is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from speech.pipeline import ReasonIQSpeechPipeline, BiasLessSpeechPipeline
from services.audio_service import audio_service
from api.rooms import router as rooms_router
from api.websocket import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load heavy speech and NLP models once at startup.
    Reused across all utterances and rooms.
    """
    print("=" * 80)
    print("ReasonIQ Server Starting: Initializing Speech & NLP Pipeline...")
    print("=" * 80)
    
    pipeline = ReasonIQSpeechPipeline(whisper_model="base")
    audio_service.set_pipeline(pipeline)
    
    print("=" * 80)
    print("ReasonIQ Speech & NLP Pipeline ready for live debate processing.")
    print("=" * 80)
    
    yield
    
    print("ReasonIQ Server Shutting Down.")


app = FastAPI(
    title="ReasonIQ Live Debate System",
    description="Real-Time Argument & Critical Thinking Analysis Engine",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(rooms_router)
app.include_router(ws_router)


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "service": "ReasonIQ Live Debate Engine",
        "pipeline_ready": audio_service.speech_pipeline is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
