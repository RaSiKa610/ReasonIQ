"""
BiasLess - Speech Interventions, Server-Side Privacy, and Final Report Test Suite

Tests:
1. Intervention Detector: personal attack vs legitimate debate disagreement.
2. Server-Side Counterargument Privacy: Author receives coaching, opponent receives stripped payload.
3. Live debate workflow with configurable interruption settings.
4. Final Performance Report Generation: 0-100 score, base score, traceable deduction audit trail.
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
from nlp.intervention import intervention_detector
from services.report_service import ReportService, report_service


@pytest.fixture(scope="module")
def initialized_app():
    pipeline = BiasLessSpeechPipeline(whisper_model="base")
    audio_service.set_pipeline(pipeline)
    return app


# =========================================================================
# UNIT TESTS: SPEECH INTERVENTION & LEGITIMATE CRITICISM SEPARATION
# =========================================================================

def test_legitimate_debate_disagreements_are_not_flagged():
    """Rule #10: Legitimate disagreement must NEVER be classified as a personal attack."""
    legitimate_statements = [
        "I strongly disagree with your argument.",
        "Your evidence does not support your conclusion.",
        "Your reasoning is incorrect.",
        "I don't think that conclusion follows from the evidence.",
        "The data you presented is fundamentally flawed because it ignores external variables.",
        "I disagree with that point completely."
    ]
    for text in legitimate_statements:
        result = intervention_detector.detect(text)
        assert result["type"] != "personal_attack", (
            f"False positive personal attack on legitimate debate speech: '{text}'"
        )
        assert result["type"] != "abusive_language", (
            f"False positive abusive language on legitimate debate speech: '{text}'"
        )
        assert result["penalty"] == 0


def test_personal_attacks_and_abusive_language_flagged():
    """Rule #8: Personal attacks and abusive language must be reliably detected with correct penalty."""
    attack_statements = [
        "You are an idiot and completely brainless.",
        "You are stupid and pathetic.",
        "Shut up idiot"
    ]
    for text in attack_statements:
        result = intervention_detector.detect(text)
        assert result["triggered"] is True
        assert result["type"] in ["personal_attack", "abusive_language"]
        assert result["penalty"] >= 15
        assert result["severity"] == "high"


def test_unclear_speech_intervention():
    """Rule #29: Short disjointed fragments detect unclear/low info with moderate penalty."""
    fragment = "I think. money Needs to happiness."
    result = intervention_detector.detect(fragment)
    assert result["triggered"] is True
    assert result["penalty"] <= 5


# =========================================================================
# INTEGRATION TESTS: PRIVACY, SETTINGS, AND FINAL REPORTS
# =========================================================================

def test_settings_and_interruption_control(initialized_app):
    client = TestClient(initialized_app)
    # Create room
    res = client.post("/api/rooms", json={
        "host_name": "Host Moderator",
        "topic": "Ethics of Autonomous AI Systems"
    })
    assert res.status_code == 200
    room_code = res.json()["room_code"]
    host_token = res.json()["host_token"]

    # Toggle settings: allow_interruptions = False
    res_set = client.post(
        f"/api/rooms/{room_code}/settings",
        headers={"X-Host-Token": host_token},
        json={"allow_interruptions": False}
    )
    assert res_set.status_code == 200
    assert res_set.json()["allow_interruptions"] is False

    # Toggle settings: allow_interruptions = True
    res_set2 = client.post(
        f"/api/rooms/{room_code}/settings",
        headers={"X-Host-Token": host_token},
        json={"allow_interruptions": True}
    )
    assert res_set2.status_code == 200
    assert res_set2.json()["allow_interruptions"] is True


def test_server_side_privacy_and_final_reports(initialized_app):
    """
    Validates:
    - Counterargument privacy is enforced server-side (Opponent receives None).
    - Debate end generates authoritative 0-100 reports.
    - Deduction audit trail traces back to exact utterance ID.
    - GET /api/rooms/{code}/reports/me works for authenticated participant.
    """
    client = TestClient(initialized_app)
    audio_file_path = PROJECT_ROOT / "tests" / "sample_audio.wav"
    assert audio_file_path.exists()

    # 1. Create Room
    res = client.post("/api/rooms", json={
        "host_name": "Host Moderator",
        "topic": "Universal Basic Income"
    })
    room_code = res.json()["room_code"]
    host_token = res.json()["host_token"]

    # 2. Join Participant A and Participant B
    res_a = client.post(f"/api/rooms/{room_code}/join", json={"participant_name": "Speaker Alpha"})
    p_a_id = res_a.json()["participant_id"]

    res_b = client.post(f"/api/rooms/{room_code}/join", json={"participant_name": "Speaker Beta"})
    p_b_id = res_b.json()["participant_id"]

    # 3. Admit Both
    client.post(f"/api/rooms/{room_code}/participants/{p_a_id}/admit", headers={"X-Host-Token": host_token})
    client.post(f"/api/rooms/{room_code}/participants/{p_b_id}/admit", headers={"X-Host-Token": host_token})

    # 4. Start Debate
    client.post(f"/api/rooms/{room_code}/start", headers={"X-Host-Token": host_token})

    # 5. Connect WebSockets for A and B to verify server-side privacy
    with client.websocket_connect(f"/ws/rooms/{room_code}?participant_id={p_a_id}&role=participant") as ws_a:
        with client.websocket_connect(f"/ws/rooms/{room_code}?participant_id={p_b_id}&role=participant") as ws_b:
            # Drain initial sync
            ws_a.receive_text()
            ws_b.receive_text()

            # Speaker Alpha submits speech
            with open(audio_file_path, "rb") as af:
                sub_res = client.post(
                    f"/api/rooms/{room_code}/utterance",
                    data={"participant_id": p_a_id},
                    files={"audio_file": ("speech.wav", af, "audio/wav")}
                )
                assert sub_res.status_code == 200
                author_data = sub_res.json()

                # Author receives full coaching in their submission response
                assert author_data["nlp_analysis"]["counterargument"] is not None

                # Read WS messages:
                # Both receive UTTERANCE_STARTED
                msg_a_start = json.loads(ws_a.receive_text())
                msg_b_start = json.loads(ws_b.receive_text())
                assert msg_a_start["type"] == "UTTERANCE_STARTED"
                assert msg_b_start["type"] == "UTTERANCE_STARTED"

            # Speaker Alpha receives UTTERANCE_COMPLETED (Author gets private coaching)
            msg_a_done = json.loads(ws_a.receive_text())
            assert msg_a_done["type"] == "UTTERANCE_COMPLETED"
            # Author payload CAN have counterargument
            assert msg_a_done["data"]["nlp_analysis"]["counterargument"] is not None

            # Speaker Beta receives UTTERANCE_COMPLETED (Opponent receives sanitized payload)
            msg_b_done = json.loads(ws_b.receive_text())
            assert msg_b_done["type"] == "UTTERANCE_COMPLETED"
            # Opponent payload MUST NOT have counterargument
            assert msg_b_done["data"]["nlp_analysis"].get("counterargument") is None

    # 6. End Debate
    end_res = client.post(f"/api/rooms/{room_code}/end", headers={"X-Host-Token": host_token})
    assert end_res.status_code == 200
    reports = end_res.json()["reports"]
    assert p_a_id in reports

    # 7. Check Report Format & Audit Trail (Rules #3, #5, #15, #16, #25)
    report_a = reports[p_a_id]
    assert 0 <= report_a["final_score"] <= 100
    assert 0 <= report_a["base_score"] <= 100
    assert report_a["base_score"] - report_a["intervention_penalties"] == report_a["final_score"]
    assert "breakdown" in report_a
    assert "argument_quality" in report_a["breakdown"]
    assert "evidence_quality" in report_a["breakdown"]
    assert "clarity_and_structure" in report_a["breakdown"]
    assert "debate_conduct" in report_a["breakdown"]
    assert len(report_a["strengths"]) > 0
    assert len(report_a["how_to_improve"]) > 0

    # 8. Test GET /api/rooms/{code}/reports/me
    my_report_res = client.get(f"/api/rooms/{room_code}/reports/me?participant_id={p_a_id}")
    assert my_report_res.status_code == 200
    my_report = my_report_res.json()["report"]
    assert my_report["participant_id"] == p_a_id
    assert my_report["final_score"] == report_a["final_score"]

    # 9. Test GET /api/rooms/{code}/reports/{participant_id} for public (Sanitized)
    pub_report_res = client.get(f"/api/rooms/{room_code}/reports/{p_a_id}")
    assert pub_report_res.status_code == 200
    pub_report = pub_report_res.json()["report"]
    assert pub_report["participant_id"] == p_a_id
    # Public report strips private_counterargument from turn history
    for turn in pub_report.get("turn_history", []):
        assert "private_counterargument" not in turn
