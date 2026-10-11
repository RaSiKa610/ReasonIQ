"""
BiasLess - Phase 1 Integration Tests

Tests all requirements:
1. Dynamic room creation & random room code uniqueness
2. Dynamic participant creation & join request
3. Host admission & dynamic speaker label assignment (Speaker A, Speaker B, etc.)
4. Room state transitions & validation
5. Test A: Direct sample_audio.wav processing through existing Speech + NLP pipeline
6. Test B: Browser-style audio conversion via PyAV adapter + Speech pipeline
7. REST API Endpoints with FastAPI TestClient
8. In-memory purity check (no database files created)
"""

import sys
import os
import wave
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# Add backend directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.room_manager import RoomManager
from services.audio_service import AudioService
from speech.pipeline import BiasLessSpeechPipeline
from main import app


@pytest.fixture(scope="module")
def shared_pipeline():
    """Load the speech pipeline once for tests."""
    pipeline = BiasLessSpeechPipeline(whisper_model="base")
    return pipeline


def test_room_manager_lifecycle():
    """Test dynamic room creation, codes, participants, and speaker labels."""
    mgr = RoomManager()

    # 1. Dynamic Room Creation
    room1 = mgr.create_room(host_name="Prof. Sharma", topic="AI in Higher Education")
    assert room1.room_code.startswith("BL")
    assert len(room1.room_code) == 6
    assert room1.status == "WAITING"
    assert room1.host_token is not None

    # Room uniqueness
    room2 = mgr.create_room(host_name="Dr. Rao", topic="Renewable Energy Policy")
    assert room1.room_code != room2.room_code

    # 2. Dynamic Join Request
    p1 = mgr.join_request(room_code=room1.room_code, participant_name="Participant One")
    assert p1.status == "WAITING"
    assert p1.speaker_label is None

    p2 = mgr.join_request(room_code=room1.room_code, participant_name="Participant Two")
    assert p2.status == "WAITING"
    assert p1.participant_id != p2.participant_id

    # Duplicate name check
    with pytest.raises(ValueError):
        mgr.join_request(room_code=room1.room_code, participant_name="Participant One")

    # 3. Host Admission & Dynamic Speaker Assignment
    admitted_p1 = mgr.admit_participant(
        room_code=room1.room_code,
        host_token=room1.host_token,
        participant_id=p1.participant_id
    )
    assert admitted_p1.status == "ADMITTED"
    assert admitted_p1.speaker_label == "Speaker A"

    admitted_p2 = mgr.admit_participant(
        room_code=room1.room_code,
        host_token=room1.host_token,
        participant_id=p2.participant_id
    )
    assert admitted_p2.status == "ADMITTED"
    assert admitted_p2.speaker_label == "Speaker B"

    # Room status transitions to READY
    assert room1.status == "READY"

    # 4. Unauthorized Host Operations
    with pytest.raises(PermissionError):
        mgr.start_debate(room_code=room1.room_code, host_token="invalid-token")

    # 5. Start Debate (READY -> LIVE)
    mgr.start_debate(room_code=room1.room_code, host_token=room1.host_token)
    assert room1.status == "LIVE"

    # 6. End Debate (LIVE -> ENDED)
    mgr.end_debate(room_code=room1.room_code, host_token=room1.host_token)
    assert room1.status == "ENDED"

    # No new audio after debate ends
    with pytest.raises(ValueError):
        mgr.create_utterance(room_code=room1.room_code, participant_id=p1.participant_id)


def test_speech_pipeline_test_a(shared_pipeline):
    """
    Test A: Verify the existing BiasLessSpeechPipeline directly processes
    sample_audio.wav -> Silero VAD -> faster-whisper -> BiasLess NLP Engine.
    """
    audio_path = PROJECT_ROOT / "tests" / "sample_audio.wav"
    assert audio_path.exists(), f"Sample audio not found at {audio_path}"

    result = shared_pipeline.process(str(audio_path))
    assert result is not None
    assert "text" in result
    assert len(result["text"]) > 0
    assert "segments" in result
    assert len(result["segments"]) > 0

    first_segment = result["segments"][0]
    assert "nlp_analysis" in first_segment
    nlp = first_segment["nlp_analysis"]
    assert nlp is not None
    assert "fallacy" in nlp
    assert "sentiment" in nlp
    assert "quality" in nlp
    assert "counterargument" in nlp


def test_pyav_audio_conversion_test_b(shared_pipeline):
    """
    Test B: Verify the PyAV adapter converts audio bytes into 16kHz WAV
    and delegates cleanly to the existing BiasLessSpeechPipeline.
    """
    audio_path = PROJECT_ROOT / "tests" / "sample_audio.wav"
    with open(audio_path, "rb") as f:
        raw_bytes = f.read()

    adapter = AudioService(speech_pipeline=shared_pipeline)
    
    # 1. Test convert_to_wav
    temp_wav = adapter.convert_to_wav(raw_bytes, target_sample_rate=16000)
    try:
        assert os.path.exists(temp_wav)
        with wave.open(temp_wav, "rb") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
    finally:
        if os.path.exists(temp_wav):
            os.unlink(temp_wav)

    # 2. Test complete process_audio via adapter
    processed = adapter.process_audio(raw_bytes)
    assert processed["status"] == "COMPLETED"
    assert len(processed["transcript"]) > 0
    assert processed["nlp_analysis"] is not None
    assert "quality" in processed["nlp_analysis"]


def test_api_rest_flow(shared_pipeline):
    """
    Test the complete REST API lifecycle using FastAPI TestClient.
    """
    from services.audio_service import audio_service
    audio_service.set_pipeline(shared_pipeline)

    client = TestClient(app)

    # 1. Health check
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["pipeline_ready"] is True

    # 2. Host creates room
    create_res = client.post("/api/rooms", json={
        "host_name": "Moderator Alice",
        "topic": "Universal Basic Income"
    })
    assert create_res.status_code == 200
    room_data = create_res.json()
    room_code = room_data["room_code"]
    host_token = room_data["host_token"]
    assert room_code.startswith("BL")

    # 3. Participant requests to join
    join_res = client.post(f"/api/rooms/{room_code}/join", json={
        "participant_name": "Bob Participant"
    })
    assert join_res.status_code == 200
    p_data = join_res.json()
    participant_id = p_data["participant_id"]

    # 4. Host admits participant
    admit_res = client.post(
        f"/api/rooms/{room_code}/participants/{participant_id}/admit",
        headers={"X-Host-Token": host_token}
    )
    assert admit_res.status_code == 200
    assert admit_res.json()["participant"]["speaker_label"] == "Speaker A"

    # 5. Host starts debate
    start_res = client.post(
        f"/api/rooms/{room_code}/start",
        headers={"X-Host-Token": host_token}
    )
    assert start_res.status_code == 200
    assert start_res.json()["status"] == "LIVE"

    # 6. Participant submits audio utterance
    audio_path = PROJECT_ROOT / "tests" / "sample_audio.wav"
    with open(audio_path, "rb") as f:
        utt_res = client.post(
            f"/api/rooms/{room_code}/utterance",
            data={"participant_id": participant_id},
            files={"audio_file": ("test.wav", f, "audio/wav")}
        )
    assert utt_res.status_code == 200
    utt_data = utt_res.json()
    assert utt_data["success"] is True
    assert utt_data["status"] == "COMPLETED"
    assert utt_data["speaker_label"] == "Speaker A"
    assert len(utt_data["transcript"]) > 0
    assert "nlp_analysis" in utt_data

    # 7. Host ends debate
    end_res = client.post(
        f"/api/rooms/{room_code}/end",
        headers={"X-Host-Token": host_token}
    )
    assert end_res.status_code == 200
    assert end_res.json()["status"] == "ENDED"

    # 8. Check room state
    get_res = client.get(f"/api/rooms/{room_code}")
    assert get_res.status_code == 200
    final_room = get_res.json()
    assert final_room["status"] == "ENDED"
    assert len(final_room["utterances"]) == 1


def test_no_database_files_created():
    """
    Verify that no database files (sqlite, db, etc.) were created during runtime.
    """
    forbidden_extensions = [".db", ".sqlite", ".sqlite3", ".rdb"]
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # ignore venv and git
        if "venv" in root or ".git" in root:
            continue
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            assert ext not in forbidden_extensions, f"Forbidden database file detected: {file} in {root}"
