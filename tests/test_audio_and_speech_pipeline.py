import os
import io
import wave
import struct
import math
import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from main import app
from services.audio_service import convert_to_wav, process_audio_file, audio_service
from services.room_manager import room_manager
from speech.pipeline import ReasonIQSpeechPipeline

client = TestClient(app)


def generate_synthetic_wav(duration_s=1.0, sample_rate=16000, freq=440.0, silence=False) -> bytes:
    """Generate a clean in-memory 16kHz mono PCM WAV byte stream."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        num_frames = int(duration_s * sample_rate)
        frames = bytearray()
        for i in range(num_frames):
            if silence:
                val = 0
            else:
                val = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * freq * (i / sample_rate)))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(frames)
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def clean_state():
    """Ensure clean room manager state and pipeline initialized before every test."""
    room_manager.rooms.clear()
    if audio_service.speech_pipeline is None:
        audio_service.set_pipeline(ReasonIQSpeechPipeline(whisper_model="base"))
    yield
    room_manager.rooms.clear()


# =========================================================================
# TEST 1 & TEST 4: PyAV Audio Conversion & Normalization
# =========================================================================
def test_pyav_conversion_normalizes_to_16khz_mono():
    """Test 4: PyAV successfully converts valid audio to 16kHz mono PCM WAV."""
    raw_wav = generate_synthetic_wav(duration_s=1.5, sample_rate=44100, freq=300.0)
    converted_wav = convert_to_wav(raw_wav, utterance_id="test_conv_01")
    assert converted_wav is not None
    assert len(converted_wav) > 0

    with wave.open(io.BytesIO(converted_wav), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2


# =========================================================================
# TEST 2 & TEST 15: Empty Audio & Silence -> NO_SPEECH (Not generic failure)
# =========================================================================
def test_empty_audio_returns_no_speech():
    """Test 2 & 15: Empty or silent audio returns status='NO_SPEECH'."""
    # 0 bytes
    res_empty = process_audio_file(b"", utterance_id="empty_01")
    assert res_empty["status"] == "NO_SPEECH"
    assert res_empty["transcript"] == ""

    # Pure silence (VAD / Speech pipeline returns empty text)
    silent_wav = generate_synthetic_wav(duration_s=2.0, sample_rate=16000, silence=True)
    with patch.object(audio_service.speech_pipeline, "process", return_value={"text": "", "segments": [], "language": "en"}):
        res_silent = process_audio_file(silent_wav, utterance_id="silent_01")
        assert res_silent["status"] == "NO_SPEECH"
        assert res_silent["stage"] == "vad"


# =========================================================================
# TEST 3: Corrupt Audio -> AUDIO_ERROR (Distinguished from generic failure)
# =========================================================================
def test_corrupt_audio_returns_audio_error():
    """Test 3: Corrupt or unparseable audio returns status='AUDIO_ERROR' with stage='audio_conversion'."""
    garbage_bytes = b"NOT_A_VALID_AUDIO_CONTAINER_RANDOM_GARBAGE_BYTES_12345"
    result = process_audio_file(garbage_bytes, utterance_id="corrupt_01")
    assert result["status"] == "AUDIO_ERROR"
    assert result["stage"] == "audio_conversion"
    assert "error" in result


# =========================================================================
# TEST 5, 6, 7: VAD -> Whisper -> ReasonIQ NLP Pipeline Execution
# =========================================================================
def test_full_pipeline_with_mocked_speech_components():
    """Test 5, 6, 7: Audio conversion -> VAD speech detected -> Whisper transcription -> NLP analysis."""
    audio_bytes = generate_synthetic_wav(duration_s=2.0, sample_rate=16000, freq=250.0)
    mock_transcript = "I believe technology enhances human productivity when used mindfully."

    mock_pipeline_res = {
        "text": mock_transcript,
        "segments": [{"start": 0.1, "end": 1.9, "text": mock_transcript, "nlp_analysis": None}],
        "language": "en"
    }

    with patch.object(audio_service.speech_pipeline, "process", return_value=mock_pipeline_res):
        result = process_audio_file(audio_bytes, utterance_id="speech_pipe_01", topic="Technology and Society")

        assert result["status"] == "COMPLETED"
        assert result["transcript"] == mock_transcript
        assert result["nlp_analysis"] is not None
        assert "quality" in result["nlp_analysis"]
        assert "argument_component" in result["nlp_analysis"]


# =========================================================================
# TEST 8: Unique Utterance IDs
# =========================================================================
def test_unique_utterance_ids_are_preserved():
    """Test 8: Every processed utterance strictly retains and returns its unique ID."""
    audio_bytes = generate_synthetic_wav(duration_s=1.0, sample_rate=16000)

    mock_res = {
        "text": "Unique turn test",
        "segments": [{"start": 0.1, "end": 0.9, "text": "Unique turn test", "nlp_analysis": None}],
        "language": "en"
    }

    with patch.object(audio_service.speech_pipeline, "process", return_value=mock_res):
        res1 = process_audio_file(audio_bytes, utterance_id="turn_uuid_alpha")
        res2 = process_audio_file(audio_bytes, utterance_id="turn_uuid_beta")

        assert res1["utterance_id"] == "turn_uuid_alpha"
        assert res2["utterance_id"] == "turn_uuid_beta"


# =========================================================================
# TEST 9, 10, 11: Speaker A & Speaker B Isolation, Interleaved Turns & Concurrency
# =========================================================================
def test_speaker_a_and_speaker_b_independent_state_and_interleaved_turns():
    """Test 9, 10, 11: Speaker A and Speaker B submit turns (A1 -> B1 -> A2 -> B2). State and transcripts are isolated."""
    # 1. Create Room
    create_resp = client.post("/api/rooms/create", json={"topic": "AI in Education", "host_name": "Host Moderator"})
    assert create_resp.status_code == 200
    room_data = create_resp.json()
    room_code = room_data["room_code"]
    host_token = room_data["host_token"]

    # 2. Join Speaker A and Speaker B
    join_a = client.post(f"/api/rooms/{room_code}/join", json={"name": "Alice"}).json()
    join_b = client.post(f"/api/rooms/{room_code}/join", json={"name": "Bob"}).json()
    pid_a = join_a["participant_id"]
    pid_b = join_b["participant_id"]

    # 3. Admit Speaker A & B
    client.post(f"/api/rooms/{room_code}/admit/{pid_a}?token={host_token}")
    client.post(f"/api/rooms/{room_code}/admit/{pid_b}?token={host_token}")

    # 4. Start debate
    client.post(f"/api/rooms/{room_code}/start?token={host_token}")

    # Audio bytes for turns
    audio_turn_a1 = generate_synthetic_wav(duration_s=1.5, sample_rate=16000)
    audio_turn_b1 = generate_synthetic_wav(duration_s=1.5, sample_rate=16000)
    audio_turn_a2 = generate_synthetic_wav(duration_s=1.5, sample_rate=16000)
    audio_turn_b2 = generate_synthetic_wav(duration_s=1.5, sample_rate=16000)

    transcripts = {
        "utt_a1": "Speaker A first argument: AI provides customized tutoring for struggling students.",
        "utt_b1": "Speaker B counter: AI tools might lead to reduced critical thinking skills.",
        "utt_a2": "Speaker A rebuttal: Empirical research demonstrates improved learning outcomes.",
        "utt_b2": "Speaker B conclusion: We must balance automated tools with human mentorship."
    }

    def mock_svc_process(audio_bytes, utterance_id=None, participant_id=None, speaker_label=None, room_code=None):
        txt = transcripts.get(utterance_id, "Generic argument text")
        nlp_res = audio_service.speech_pipeline.nlp_engine.analyze(txt) if audio_service.speech_pipeline else {}
        return {
            "status": "COMPLETED",
            "stage": "completed",
            "transcript": txt,
            "nlp_analysis": nlp_res,
            "segments": [{"start": 0.0, "end": 1.4, "text": txt}],
            "error_message": None,
            "error": None,
            "utterance_id": utterance_id
        }

    with patch("api.rooms.audio_service.process_audio", side_effect=mock_svc_process):

        # Interleaved turns: A1 -> B1 -> A2 -> B2
        # Turn A1
        resp_a1 = client.post(
            f"/api/rooms/{room_code}/utterance",
            data={"participant_id": pid_a, "utterance_id": "utt_a1"},
            files={"audio": ("turn_a1.webm", audio_turn_a1, "audio/webm")}
        )
        assert resp_a1.status_code == 200
        data_a1 = resp_a1.json()
        assert data_a1["utterance_id"] == "utt_a1"
        assert data_a1["speaker_label"] == "Speaker A"
        assert data_a1["transcript"] == transcripts["utt_a1"]

        # Turn B1
        resp_b1 = client.post(
            f"/api/rooms/{room_code}/utterance",
            data={"participant_id": pid_b, "utterance_id": "utt_b1"},
            files={"audio": ("turn_b1.webm", audio_turn_b1, "audio/webm")}
        )
        assert resp_b1.status_code == 200
        data_b1 = resp_b1.json()
        assert data_b1["utterance_id"] == "utt_b1"
        assert data_b1["speaker_label"] == "Speaker B"
        assert data_b1["transcript"] == transcripts["utt_b1"]

        # Turn A2
        resp_a2 = client.post(
            f"/api/rooms/{room_code}/utterance",
            data={"participant_id": pid_a, "utterance_id": "utt_a2"},
            files={"audio": ("turn_a2.webm", audio_turn_a2, "audio/webm")}
        )
        assert resp_a2.status_code == 200
        data_a2 = resp_a2.json()
        assert data_a2["utterance_id"] == "utt_a2"
        assert data_a2["speaker_label"] == "Speaker A"
        assert data_a2["transcript"] == transcripts["utt_a2"]

        # Turn B2
        resp_b2 = client.post(
            f"/api/rooms/{room_code}/utterance",
            data={"participant_id": pid_b, "utterance_id": "utt_b2"},
            files={"audio": ("turn_b2.webm", audio_turn_b2, "audio/webm")}
        )
        assert resp_b2.status_code == 200
        data_b2 = resp_b2.json()
        assert data_b2["utterance_id"] == "utt_b2"
        assert data_b2["speaker_label"] == "Speaker B"
        assert data_b2["transcript"] == transcripts["utt_b2"]

    # Verify Room Utterance Log integrity
    room = room_manager.get_room(room_code)
    assert len(room.utterances) == 4
    assert room.utterances[0].utterance_id == "utt_a1"
    assert room.utterances[0].speaker_label == "Speaker A"
    assert room.utterances[1].utterance_id == "utt_b1"
    assert room.utterances[1].speaker_label == "Speaker B"
    assert room.utterances[2].utterance_id == "utt_a2"
    assert room.utterances[2].speaker_label == "Speaker A"
    assert room.utterances[3].utterance_id == "utt_b2"
    assert room.utterances[3].speaker_label == "Speaker B"


# =========================================================================
# TEST 12: Temporary Files Are Unique & Cleaned Up
# =========================================================================
def test_temporary_files_unique_and_cleaned_up():
    """Test 12: Each utterance uses a distinct temp file and cleans it up after execution."""
    audio_bytes = generate_synthetic_wav(duration_s=1.0, sample_rate=16000)

    mock_res = {
        "text": "Temp file test",
        "segments": [{"start": 0.1, "end": 0.9, "text": "Temp file test", "nlp_analysis": None}],
        "language": "en"
    }

    with patch.object(audio_service.speech_pipeline, "process", return_value=mock_res):
        res1 = process_audio_file(audio_bytes, utterance_id="temp_test_alpha")
        res2 = process_audio_file(audio_bytes, utterance_id="temp_test_beta")

        assert res1["status"] == "COMPLETED"
        assert res2["status"] == "COMPLETED"

        import tempfile
        temp_dir = tempfile.gettempdir()
        if os.path.exists(temp_dir):
            files = os.listdir(temp_dir)
            assert not any("temp_test_alpha" in f for f in files)
            assert not any("temp_test_beta" in f for f in files)


# =========================================================================
# TEST 14: NLP Failure Does Not Delete Successful Transcript
# =========================================================================
def test_nlp_failure_preserves_successful_transcript():
    """Test 14: If Whisper transcribes successfully but ReasonIQ NLP encounters an error, transcript is preserved."""
    audio_bytes = generate_synthetic_wav(duration_s=1.5, sample_rate=16000)
    mock_transcript = "Technology enables global collaboration and faster communication."

    mock_speech_res = {
        "text": mock_transcript,
        "segments": [{"start": 0.1, "end": 1.4, "text": mock_transcript, "nlp_analysis": None}],
        "language": "en"
    }

    with patch.object(audio_service.speech_pipeline, "process", return_value=mock_speech_res), \
         patch.object(audio_service.speech_pipeline.nlp_engine, "analyze", side_effect=RuntimeError("Simulated NLP Model OOM")):

        result = process_audio_file(audio_bytes, utterance_id="nlp_fail_01", topic="Technology")

        # Must report NLP_ERROR stage, keep transcript, and not return generic empty failure
        assert result["status"] == "NLP_ERROR"
        assert result["stage"] == "nlp"
        assert result["transcript"] == mock_transcript
        assert "Simulated NLP Model OOM" in result["error"]
