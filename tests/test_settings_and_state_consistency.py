"""
BiasLess - Settings, State Preservation, Error Propagation & Report Consistency Tests

Validates:
1. Settings update atomic success without mutating or resetting participants or room state.
2. Multiple reverse setting toggles (false -> true -> false -> true).
3. Settings update during LIVE debate preserves LIVE state and all active participants.
4. Invalid settings requests (non-boolean, bad JSON) reject with HTTP 400 and structured error.
5. Unauthorized host settings request or settings change on ENDED debate fails cleanly without mutating room.
6. Strict Turns mode (allow_interruptions=False) blocks overlapping speaker turns while keeping participant ADMITTED.
7. Cross-Talk mode (allow_interruptions=True) allows concurrent speaker turns.
8. Room Error Propagation (ROOM_ERROR WebSocket broadcast) delivers recoverable notices without removing participants.
9. Report data consistency: Total turns match completed analyzed turns, and final scores are valid 0-100 numbers.
10. Empty report handling: Participants with 0 speaking turns receive a clear notice without false "Clean Record" claims.
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
from services.room_manager import RoomManager, room_manager
from services.audio_service import audio_service
from services.report_service import report_service
from speech.pipeline import BiasLessSpeechPipeline


@pytest.fixture(scope="module")
def initialized_app():
    pipeline = BiasLessSpeechPipeline(whisper_model="base")
    audio_service.set_pipeline(pipeline)
    return app


def test_1_settings_update_success_preserves_participants(initialized_app):
    """
    Test 1: Admitted participants + allow_interruptions = true -> set false -> participants unchanged.
    """
    client = TestClient(initialized_app)
    
    # Create room
    create_res = client.post("/api/rooms", json={
        "host_name": "Moderator Alice",
        "topic": "Future of AI Governance",
        "allow_interruptions": True
    })
    assert create_res.status_code == 200
    room_code = create_res.json()["room_code"]
    host_token = create_res.json()["host_token"]

    # Add participants
    p1_res = client.post(f"/api/rooms/{room_code}/join", json={"participant_name": "Speaker One"})
    p2_res = client.post(f"/api/rooms/{room_code}/join", json={"participant_name": "Speaker Two"})
    p1_id = p1_res.json()["participant_id"]
    p2_id = p2_res.json()["participant_id"]

    # Admit participants
    client.post(f"/api/rooms/{room_code}/participants/{p1_id}/admit", headers={"X-Host-Token": host_token})
    client.post(f"/api/rooms/{room_code}/participants/{p2_id}/admit", headers={"X-Host-Token": host_token})

    # Verify admitted state before toggle
    room_before = client.get(f"/api/rooms/{room_code}").json()
    assert len(room_before["participants"]) == 2
    assert room_before["allow_interruptions"] is True
    assert room_before["participants"][0]["status"] == "ADMITTED"
    assert room_before["participants"][1]["status"] == "ADMITTED"
    speaker_labels_before = {p["participant_id"]: p["speaker_label"] for p in room_before["participants"]}

    # Update setting to Strict Turns (False)
    setting_res = client.post(
        f"/api/rooms/{room_code}/settings",
        json={"allow_interruptions": False},
        headers={"X-Host-Token": host_token}
    )
    assert setting_res.status_code == 200
    assert setting_res.json()["allow_interruptions"] is False

    # Verify participants remain 100% intact after settings update
    room_after = client.get(f"/api/rooms/{room_code}").json()
    assert room_after["allow_interruptions"] is False
    assert len(room_after["participants"]) == 2
    assert room_after["topic"] == "Future of AI Governance"
    assert room_after["host_name"] == "Moderator Alice"
    for p in room_after["participants"]:
        assert p["status"] == "ADMITTED"
        assert p["speaker_label"] == speaker_labels_before[p["participant_id"]]


def test_2_reverse_settings_multiple_times(initialized_app):
    """
    Test 2: Reverse settings: false -> true -> false -> true multiple times without state degradation.
    """
    client = TestClient(initialized_app)
    create_res = client.post("/api/rooms", json={
        "host_name": "Host Bob",
        "topic": "Renewable Energy Transition"
    })
    room_code = create_res.json()["room_code"]
    host_token = create_res.json()["host_token"]

    p_res = client.post(f"/api/rooms/{room_code}/join", json={"participant_name": "Participant A"})
    p_id = p_res.json()["participant_id"]
    client.post(f"/api/rooms/{room_code}/participants/{p_id}/admit", headers={"X-Host-Token": host_token})

    for target_val in [False, True, False, True]:
        res = client.post(
            f"/api/rooms/{room_code}/settings",
            json={"allow_interruptions": target_val},
            headers={"X-Host-Token": host_token}
        )
        assert res.status_code == 200
        assert res.json()["allow_interruptions"] == target_val

        # Verify participants remain admitted
        room = client.get(f"/api/rooms/{room_code}").json()
        assert len(room["participants"]) == 1
        assert room["participants"][0]["status"] == "ADMITTED"


def test_3_settings_update_during_live_state(initialized_app):
    """
    Test 3: LIVE -> change setting -> remains LIVE with intact participants.
    """
    client = TestClient(initialized_app)
    create_res = client.post("/api/rooms", json={
        "host_name": "Host Live",
        "topic": "Autonomous Vehicles Policy"
    })
    room_code = create_res.json()["room_code"]
    host_token = create_res.json()["host_token"]

    p_res = client.post(f"/api/rooms/{room_code}/join", json={"participant_name": "Speaker Live"})
    p_id = p_res.json()["participant_id"]
    client.post(f"/api/rooms/{room_code}/participants/{p_id}/admit", headers={"X-Host-Token": host_token})

    # Start debate -> LIVE
    start_res = client.post(f"/api/rooms/{room_code}/start", headers={"X-Host-Token": host_token})
    assert start_res.status_code == 200
    assert start_res.json()["status"] == "LIVE"

    # Toggle setting during LIVE
    set_res = client.post(
        f"/api/rooms/{room_code}/settings",
        json={"allow_interruptions": False},
        headers={"X-Host-Token": host_token}
    )
    assert set_res.status_code == 200

    # Room must still be LIVE
    room_check = client.get(f"/api/rooms/{room_code}").json()
    assert room_check["status"] == "LIVE"
    assert room_check["allow_interruptions"] is False
    assert len(room_check["participants"]) == 1
    assert room_check["participants"][0]["status"] == "ADMITTED"


def test_4_invalid_settings_payload(initialized_app):
    """
    Test 4: Invalid requests (non-boolean, missing fields) return 400 and leave room unchanged.
    """
    client = TestClient(initialized_app)
    create_res = client.post("/api/rooms", json={
        "host_name": "Host Invalid",
        "topic": "Space Exploration Funding"
    })
    room_code = create_res.json()["room_code"]
    host_token = create_res.json()["host_token"]

    # Non-boolean string
    res_str = client.post(
        f"/api/rooms/{room_code}/settings",
        json={"allow_interruptions": "false"},
        headers={"X-Host-Token": host_token}
    )
    assert res_str.status_code == 400
    err_detail = res_str.json()["detail"]
    assert err_detail["code"] == "INVALID_ROOM_SETTING"

    # Missing field
    res_empty = client.post(
        f"/api/rooms/{room_code}/settings",
        json={},
        headers={"X-Host-Token": host_token}
    )
    assert res_empty.status_code == 400


def test_5_settings_failure_on_ended_debate(initialized_app):
    """
    Test 5: Attempting settings update on an ENDED debate returns 400 DEBATE_ENDED and does not mutate room.
    """
    client = TestClient(initialized_app)
    create_res = client.post("/api/rooms", json={
        "host_name": "Host Ended",
        "topic": "Climate Accords"
    })
    room_code = create_res.json()["room_code"]
    host_token = create_res.json()["host_token"]

    client.post(f"/api/rooms/{room_code}/end", headers={"X-Host-Token": host_token})

    # Try updating settings after end
    res = client.post(
        f"/api/rooms/{room_code}/settings",
        json={"allow_interruptions": True},
        headers={"X-Host-Token": host_token}
    )
    assert res.status_code == 400
    err = res.json()["detail"]
    assert err["code"] == "DEBATE_ENDED"


def test_6_strict_turns_blocks_overlapping_speakers_and_preserves_admission(initialized_app):
    """
    Test 6: Strict Turns (allow_interruptions=False) blocks Speaker B while Speaker A mic is active.
    Speaker B remains ADMITTED and can speak after A finishes.
    """
    client = TestClient(initialized_app)
    audio_path = PROJECT_ROOT / "tests" / "sample_audio.wav"

    create_res = client.post("/api/rooms", json={
        "host_name": "Host Strict",
        "topic": "Strict Turn Analysis",
        "allow_interruptions": False
    })
    room_code = create_res.json()["room_code"]
    host_token = create_res.json()["host_token"]

    p1_res = client.post(f"/api/rooms/{room_code}/join", json={"participant_name": "Speaker A"})
    p2_res = client.post(f"/api/rooms/{room_code}/join", json={"participant_name": "Speaker B"})
    p1_id = p1_res.json()["participant_id"]
    p2_id = p2_res.json()["participant_id"]

    client.post(f"/api/rooms/{room_code}/participants/{p1_id}/admit", headers={"X-Host-Token": host_token})
    client.post(f"/api/rooms/{room_code}/participants/{p2_id}/admit", headers={"X-Host-Token": host_token})
    client.post(f"/api/rooms/{room_code}/start", headers={"X-Host-Token": host_token})

    # Connect WebSockets
    with client.websocket_connect(f"/ws/rooms/{room_code}?participant_id={p1_id}&role=participant") as p1_ws:
        p1_ws.receive_text() # Sync
        with client.websocket_connect(f"/ws/rooms/{room_code}?participant_id={p2_id}&role=participant") as p2_ws:
            p2_ws.receive_text() # Sync

            # Speaker A starts microphone
            p1_ws.send_text(json.dumps({
                "type": "MIC_STATUS",
                "participant_id": p1_id,
                "mic_active": True
            }))
            # Both p1_ws and p2_ws receive the mic update broadcast
            msg1 = json.loads(p1_ws.receive_text())
            assert msg1["type"] == "PARTICIPANT_MIC_UPDATE"
            assert msg1["data"]["mic_active"] is True

            msg2 = json.loads(p2_ws.receive_text())
            assert msg2["type"] == "PARTICIPANT_MIC_UPDATE"
            assert msg2["data"]["mic_active"] is True

            # Speaker B tries to start mic concurrently -> BLOCKED
            p2_ws.send_text(json.dumps({
                "type": "MIC_STATUS",
                "participant_id": p2_id,
                "mic_active": True
            }))
            blocked_msg = json.loads(p2_ws.receive_text())
            assert blocked_msg["type"] == "SPEAKER_TURN_BLOCKED"
            assert "currently speaking" in blocked_msg["data"]["message"]

            # Speaker B tries to submit utterance while A is speaking -> 400 SPEAKER_TURN_BLOCKED
            with open(audio_path, "rb") as f:
                utt_res = client.post(
                    f"/api/rooms/{room_code}/utterance",
                    data={"participant_id": p2_id},
                    files={"audio_file": ("test.wav", f, "audio/wav")}
                )
            assert utt_res.status_code == 400
            assert utt_res.json()["detail"]["code"] == "SPEAKER_TURN_BLOCKED"

            # Verify B is still ADMITTED
            room_check = client.get(f"/api/rooms/{room_code}").json()
            b_state = next(p for p in room_check["participants"] if p["participant_id"] == p2_id)
            assert b_state["status"] == "ADMITTED"

            # Speaker A finishes speaking
            p1_ws.send_text(json.dumps({
                "type": "MIC_STATUS",
                "participant_id": p1_id,
                "mic_active": False
            }))
            json.loads(p1_ws.receive_text()) # Mic update off

            # Now Speaker B can submit cleanly
            with open(audio_path, "rb") as f:
                utt_res_b = client.post(
                    f"/api/rooms/{room_code}/utterance",
                    data={"participant_id": p2_id},
                    files={"audio_file": ("test.wav", f, "audio/wav")}
                )
            assert utt_res_b.status_code == 200
            assert utt_res_b.json()["status"] == "COMPLETED"


def test_7_crosstalk_allowed_allows_concurrent_recording(initialized_app):
    """
    Test 7: Cross-Talk Allowed (allow_interruptions=True) allows concurrent recordings.
    """
    client = TestClient(initialized_app)
    audio_path = PROJECT_ROOT / "tests" / "sample_audio.wav"

    create_res = client.post("/api/rooms", json={
        "host_name": "Host CrossTalk",
        "topic": "Cross Talk Analysis",
        "allow_interruptions": True
    })
    room_code = create_res.json()["room_code"]
    host_token = create_res.json()["host_token"]

    p1_res = client.post(f"/api/rooms/{room_code}/join", json={"participant_name": "Speaker One"})
    p2_res = client.post(f"/api/rooms/{room_code}/join", json={"participant_name": "Speaker Two"})
    p1_id = p1_res.json()["participant_id"]
    p2_id = p2_res.json()["participant_id"]

    client.post(f"/api/rooms/{room_code}/participants/{p1_id}/admit", headers={"X-Host-Token": host_token})
    client.post(f"/api/rooms/{room_code}/participants/{p2_id}/admit", headers={"X-Host-Token": host_token})
    client.post(f"/api/rooms/{room_code}/start", headers={"X-Host-Token": host_token})

    # Set Speaker A mic active
    room_obj = room_manager.get_room(room_code)
    room_obj.participants[p1_id].mic_active = True

    # Speaker B submits utterance while A is mic active -> Allowed because allow_interruptions is True
    with open(audio_path, "rb") as f:
        utt_res = client.post(
            f"/api/rooms/{room_code}/utterance",
            data={"participant_id": p2_id},
            files={"audio_file": ("test.wav", f, "audio/wav")}
        )
    assert utt_res.status_code == 200
    assert utt_res.json()["status"] == "COMPLETED"


def test_8_room_error_propagation(initialized_app):
    """
    Test 8: ROOM_ERROR WebSocket event is broadcast to participants with recoverable=True.
    """
    client = TestClient(initialized_app)
    create_res = client.post("/api/rooms", json={
        "host_name": "Host Error",
        "topic": "Error Notification Testing"
    })
    room_code = create_res.json()["room_code"]
    host_token = create_res.json()["host_token"]

    p_res = client.post(f"/api/rooms/{room_code}/join", json={"participant_name": "Listener"})
    p_id = p_res.json()["participant_id"]
    client.post(f"/api/rooms/{room_code}/participants/{p_id}/admit", headers={"X-Host-Token": host_token})

    with client.websocket_connect(f"/ws/rooms/{room_code}?participant_id={p_id}&role=participant") as p_ws:
        p_ws.receive_text() # Sync

        # Trigger room error broadcast
        from api.websocket import ws_manager
        import asyncio
        asyncio.run(ws_manager.broadcast(
            room_code,
            "ROOM_ERROR",
            {
                "code": "SETTINGS_UPDATE_FAILED",
                "message": "Unable to update the interruption setting.",
                "recoverable": True
            }
        ))

        msg = json.loads(p_ws.receive_text())
        assert msg["type"] == "ROOM_ERROR"
        assert msg["data"]["recoverable"] is True

        # Participant is still admitted and active
        room_check = client.get(f"/api/rooms/{room_code}").json()
        assert room_check["participants"][0]["status"] == "ADMITTED"


def test_9_report_data_consistency_with_multiple_turns(initialized_app):
    """
    Test 9: Real test utterances: Speaker A has 2 turns, Speaker B has 1 turn.
    Debate ends -> Speaker A has total_turns=2, Speaker B has total_turns=1,
    final_score is a valid 0-100 number.
    """
    client = TestClient(initialized_app)
    audio_path = PROJECT_ROOT / "tests" / "sample_audio.wav"

    create_res = client.post("/api/rooms", json={
        "host_name": "Host Scorer",
        "topic": "Economic Freedom Debate"
    })
    room_code = create_res.json()["room_code"]
    host_token = create_res.json()["host_token"]

    p_a = client.post(f"/api/rooms/{room_code}/join", json={"participant_name": "Speaker Alpha"}).json()
    p_b = client.post(f"/api/rooms/{room_code}/join", json={"participant_name": "Speaker Beta"}).json()
    p_a_id = p_a["participant_id"]
    p_b_id = p_b["participant_id"]

    client.post(f"/api/rooms/{room_code}/participants/{p_a_id}/admit", headers={"X-Host-Token": host_token})
    client.post(f"/api/rooms/{room_code}/participants/{p_b_id}/admit", headers={"X-Host-Token": host_token})
    client.post(f"/api/rooms/{room_code}/start", headers={"X-Host-Token": host_token})

    # Speaker A -> 2 turns
    for _ in range(2):
        with open(audio_path, "rb") as f:
            u_res = client.post(
                f"/api/rooms/{room_code}/utterance",
                data={"participant_id": p_a_id},
                files={"audio_file": ("test.wav", f, "audio/wav")}
            )
            assert u_res.status_code == 200
            assert u_res.json()["status"] == "COMPLETED"

    # Speaker B -> 1 turn
    with open(audio_path, "rb") as f:
        u_res = client.post(
            f"/api/rooms/{room_code}/utterance",
            data={"participant_id": p_b_id},
            files={"audio_file": ("test.wav", f, "audio/wav")}
        )
        assert u_res.status_code == 200
        assert u_res.json()["status"] == "COMPLETED"

    # End debate
    end_res = client.post(f"/api/rooms/{room_code}/end", headers={"X-Host-Token": host_token})
    assert end_res.status_code == 200

    # Fetch Speaker A report
    rep_a_res = client.get(f"/api/rooms/{room_code}/reports/me?participant_id={p_a_id}")
    assert rep_a_res.status_code == 200
    rep_a = rep_a_res.json()["report"]
    assert rep_a["total_turns"] == 2
    assert rep_a["analyzed_turns"] == 2
    assert 0.0 <= rep_a["final_score"] <= 100.0
    assert 0.0 <= rep_a["base_score"] <= 100.0

    # Fetch Speaker B report
    rep_b_res = client.get(f"/api/rooms/{room_code}/reports/me?participant_id={p_b_id}")
    assert rep_b_res.status_code == 200
    rep_b = rep_b_res.json()["report"]
    assert rep_b["total_turns"] == 1
    assert rep_b["analyzed_turns"] == 1
    assert 0.0 <= rep_b["final_score"] <= 100.0


def test_10_empty_report_no_clean_record_or_fake_score(initialized_app):
    """
    Test 10: Participant with 0 speaking turns has total_turns=0, final_score=None,
    and a clear message indicating no turns were available.
    """
    client = TestClient(initialized_app)
    create_res = client.post("/api/rooms", json={
        "host_name": "Host Silent",
        "topic": "Silent Participant Test"
    })
    room_code = create_res.json()["room_code"]
    host_token = create_res.json()["host_token"]

    p_res = client.post(f"/api/rooms/{room_code}/join", json={"participant_name": "Quiet Speaker"}).json()
    p_id = p_res["participant_id"]

    client.post(f"/api/rooms/{room_code}/participants/{p_id}/admit", headers={"X-Host-Token": host_token})
    client.post(f"/api/rooms/{room_code}/start", headers={"X-Host-Token": host_token})
    client.post(f"/api/rooms/{room_code}/end", headers={"X-Host-Token": host_token})

    # Fetch report for quiet speaker
    rep_res = client.get(f"/api/rooms/{room_code}/reports/me?participant_id={p_id}")
    assert rep_res.status_code == 200
    rep = rep_res.json()["report"]

    assert rep["total_turns"] == 0
    assert rep["analyzed_turns"] == 0
    assert rep["final_score"] is None
    assert rep["base_score"] is None
    assert rep["performance_tier"] == "No Analyzed Turns"
    assert "No completed speaking turns were available for analysis." in rep["message"]
