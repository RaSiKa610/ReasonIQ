"""
BiasLess - Live Multi-Client WebSocket & Multi-Participant Flow Test

Simulates real concurrent Host, Participant A, and Participant B sessions
interacting with the live FastAPI application over WebSockets and REST.
Tests real speech processing (Whisper), real NLP (BiasLessNLPEngine),
and verifies speaker separation (Speaker A vs Speaker B).
"""

import sys
import os
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from main import app
from services.audio_service import audio_service
from speech.pipeline import BiasLessSpeechPipeline


@pytest.fixture(scope="module")
def initialized_app():
    pipeline = BiasLessSpeechPipeline(whisper_model="base")
    audio_service.set_pipeline(pipeline)
    return app


def test_full_multiclient_live_debate(initialized_app):
    client = TestClient(initialized_app)
    audio_file_path = PROJECT_ROOT / "tests" / "sample_audio.wav"
    assert audio_file_path.exists()

    # ----------------------------------------------------
    # 1. Host creates room
    # ----------------------------------------------------
    res = client.post("/api/rooms", json={
        "host_name": "Moderator Alice",
        "topic": "Should AI be regulated by international law?"
    })
    assert res.status_code == 200
    host_data = res.json()
    room_code = host_data["room_code"]
    host_token = host_data["host_token"]
    assert room_code.startswith("BL")
    assert len(room_code) == 6

    # ----------------------------------------------------
    # 2. Open WebSockets for Host, Participant A, Participant B
    # ----------------------------------------------------
    with client.websocket_connect(f"/ws/rooms/{room_code}?role=host") as host_ws:
        # Host receives initial ROOM_SYNC
        msg = json.loads(host_ws.receive_text())
        assert msg["type"] == "ROOM_SYNC"
        assert msg["data"]["room_code"] == room_code
        assert msg["data"]["status"] == "WAITING"

        # ------------------------------------------------
        # 3. Participant A requests to join
        # ------------------------------------------------
        join_a = client.post(f"/api/rooms/{room_code}/join", json={"participant_name": "Rahul"})
        assert join_a.status_code == 200
        p_a_id = join_a.json()["participant_id"]

        # Host receives NEW_JOIN_REQUEST
        msg = json.loads(host_ws.receive_text())
        assert msg["type"] == "NEW_JOIN_REQUEST"
        assert msg["data"]["participant_id"] == p_a_id
        assert msg["data"]["name"] == "Rahul"

        # Participant A connects to WS
        with client.websocket_connect(f"/ws/rooms/{room_code}?participant_id={p_a_id}&role=participant") as p_a_ws:
            p_a_sync = json.loads(p_a_ws.receive_text())
            assert p_a_sync["type"] == "ROOM_SYNC"

            # --------------------------------------------
            # 4. Participant B requests to join
            # --------------------------------------------
            join_b = client.post(f"/api/rooms/{room_code}/join", json={"participant_name": "Priya"})
            assert join_b.status_code == 200
            p_b_id = join_b.json()["participant_id"]
            assert p_a_id != p_b_id

            # Host receives NEW_JOIN_REQUEST for Priya
            msg = json.loads(host_ws.receive_text())
            assert msg["type"] == "NEW_JOIN_REQUEST"
            assert msg["data"]["name"] == "Priya"

            # Participant A also receives the broadcast
            msg = json.loads(p_a_ws.receive_text())
            assert msg["type"] == "NEW_JOIN_REQUEST"

            # Participant B connects to WS
            with client.websocket_connect(f"/ws/rooms/{room_code}?participant_id={p_b_id}&role=participant") as p_b_ws:
                p_b_sync = json.loads(p_b_ws.receive_text())
                assert p_b_sync["type"] == "ROOM_SYNC"

                # ----------------------------------------
                # 5. Host admits Participant A
                # ----------------------------------------
                admit_a = client.post(
                    f"/api/rooms/{room_code}/participants/{p_a_id}/admit",
                    headers={"X-Host-Token": host_token}
                )
                assert admit_a.status_code == 200
                assert admit_a.json()["participant"]["speaker_label"] == "Speaker A"

                # Host receives PARTICIPANT_ADMITTED
                msg_host = json.loads(host_ws.receive_text())
                assert msg_host["type"] == "PARTICIPANT_ADMITTED"
                assert msg_host["data"]["speaker_label"] == "Speaker A"

                # Participant A receives PARTICIPANT_ADMITTED and YOU_ARE_ADMITTED
                msg_pa1 = json.loads(p_a_ws.receive_text())
                msg_pa2 = json.loads(p_a_ws.receive_text())
                pa_types = {msg_pa1["type"], msg_pa2["type"]}
                assert "PARTICIPANT_ADMITTED" in pa_types
                assert "YOU_ARE_ADMITTED" in pa_types

                # Participant B receives PARTICIPANT_ADMITTED
                msg_pb = json.loads(p_b_ws.receive_text())
                assert msg_pb["type"] == "PARTICIPANT_ADMITTED"

                # ----------------------------------------
                # 6. Host admits Participant B
                # ----------------------------------------
                admit_b = client.post(
                    f"/api/rooms/{room_code}/participants/{p_b_id}/admit",
                    headers={"X-Host-Token": host_token}
                )
                assert admit_b.status_code == 200
                assert admit_b.json()["participant"]["speaker_label"] == "Speaker B"

                # Drain admission events
                json.loads(host_ws.receive_text())
                json.loads(p_a_ws.receive_text())
                json.loads(p_b_ws.receive_text())
                json.loads(p_b_ws.receive_text())

                # ----------------------------------------
                # 7. Host starts debate
                # ----------------------------------------
                start_res = client.post(
                    f"/api/rooms/{room_code}/start",
                    headers={"X-Host-Token": host_token}
                )
                assert start_res.status_code == 200
                assert start_res.json()["status"] == "LIVE"

                # All clients receive DEBATE_STARTED
                assert json.loads(host_ws.receive_text())["type"] == "DEBATE_STARTED"
                assert json.loads(p_a_ws.receive_text())["type"] == "DEBATE_STARTED"
                assert json.loads(p_b_ws.receive_text())["type"] == "DEBATE_STARTED"

                # ----------------------------------------
                # 8. Participant A speaks (real audio)
                # ----------------------------------------
                with open(audio_file_path, "rb") as f:
                    utt_a_res = client.post(
                        f"/api/rooms/{room_code}/utterance",
                        data={"participant_id": p_a_id},
                        files={"audio_file": ("utt1.wav", f, "audio/wav")}
                    )
                assert utt_a_res.status_code == 200
                utt_a_data = utt_a_res.json()
                assert utt_a_data["success"] is True
                assert utt_a_data["status"] == "COMPLETED"
                assert utt_a_data["speaker_label"] == "Speaker A"
                assert utt_a_data["speaker_name"] == "Rahul"
                assert len(utt_a_data["transcript"]) > 0
                assert utt_a_data["nlp_analysis"] is not None

                # Verify WebSocket events for Utterance A (STARTED + COMPLETED)
                for ws in [host_ws, p_a_ws, p_b_ws]:
                    ev1 = json.loads(ws.receive_text())
                    ev2 = json.loads(ws.receive_text())
                    assert ev1["type"] == "UTTERANCE_STARTED"
                    assert ev2["type"] == "UTTERANCE_COMPLETED"
                    assert ev2["data"]["speaker_label"] == "Speaker A"
                    assert ev2["data"]["speaker_name"] == "Rahul"

                # ----------------------------------------
                # 9. Participant B speaks (real audio)
                # ----------------------------------------
                with open(audio_file_path, "rb") as f:
                    utt_b_res = client.post(
                        f"/api/rooms/{room_code}/utterance",
                        data={"participant_id": p_b_id},
                        files={"audio_file": ("utt2.wav", f, "audio/wav")}
                    )
                assert utt_b_res.status_code == 200
                utt_b_data = utt_b_res.json()
                assert utt_b_data["success"] is True
                assert utt_b_data["status"] == "COMPLETED"
                assert utt_b_data["speaker_label"] == "Speaker B"
                assert utt_b_data["speaker_name"] == "Priya"

                # Verify WebSocket events for Utterance B
                for ws in [host_ws, p_a_ws, p_b_ws]:
                    ev1 = json.loads(ws.receive_text())
                    ev2 = json.loads(ws.receive_text())
                    assert ev1["type"] == "UTTERANCE_STARTED"
                    assert ev2["type"] == "UTTERANCE_COMPLETED"
                    assert ev2["data"]["speaker_label"] == "Speaker B"
                    assert ev2["data"]["speaker_name"] == "Priya"

                # ----------------------------------------
                # 10. No-Speech case: empty/tiny audio
                # ----------------------------------------
                fake_silent_bytes = b"RIFF....WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
                no_speech_res = client.post(
                    f"/api/rooms/{room_code}/utterance",
                    data={"participant_id": p_a_id},
                    files={"audio_file": ("silent.wav", fake_silent_bytes, "audio/wav")}
                )
                assert no_speech_res.status_code == 200
                assert no_speech_res.json()["status"] == "NO_SPEECH"

                # ----------------------------------------
                # 11. Host ends debate
                # ----------------------------------------
                end_res = client.post(
                    f"/api/rooms/{room_code}/end",
                    headers={"X-Host-Token": host_token}
                )
                assert end_res.status_code == 200
                assert end_res.json()["status"] == "ENDED"

                # ----------------------------------------
                # 12. Audio submission after debate ends is rejected
                # ----------------------------------------
                with open(audio_file_path, "rb") as f:
                    after_end_res = client.post(
                        f"/api/rooms/{room_code}/utterance",
                        data={"participant_id": p_a_id},
                        files={"audio_file": ("after.wav", f, "audio/wav")}
                    )
                assert after_end_res.status_code == 400
                assert "not LIVE" in after_end_res.json()["detail"]
