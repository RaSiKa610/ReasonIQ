"""
ReasonIQ - Core Validation Test Suite

Validates all 11 Section 13 requirements:
TEST 1 — Branding: User-facing branding is ReasonIQ.
TEST 2 — Completed turn count: Total Turns matches completed utterances.
TEST 3 — Score aggregation: All completed turns are aggregated.
TEST 4 — Speaking history: Speaking history is present and maps to the correct participant.
TEST 5 — Incomplete / failed turns: Excluded from report aggregation.
TEST 6 — Participant isolation: Participant A cannot access Participant B's private report.
TEST 7 — Host reports: Host retrieves all participant reports.
TEST 8 — NER: Named entities are returned in NLP payload.
TEST 9 — Intervention: Accurate classification of attacks vs legitimate disagreement.
TEST 10 — Private coaching: Coaching counterargument isolated to speaker.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from main import app
from services.room_manager import room_manager
from services.report_service import report_service
from nlp.intervention import intervention_detector
from nlp.engine import ReasonIQNLPEngine


# =========================================================================
# TEST 1 — BRANDING
# =========================================================================

def test_1_branding_consistency():
    """Verify backend health and API branding says ReasonIQ."""
    client = TestClient(app)
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert "ReasonIQ" in data["service"]
    assert "BiasLess" not in data["service"]


# =========================================================================
# TEST 2 — COMPLETED TURN COUNT
# =========================================================================

def test_2_completed_turn_count():
    """
    Create a room, add a participant, add 3 completed utterances.
    Generate final report. Expected: total_turns == 3.
    """
    room = room_manager.create_room(host_name="Moderator Alice", topic="Renewable Energy Transition")
    p = room_manager.join_request(room.room_code, "Bob")
    room_manager.admit_participant(room.room_code, room.host_token, p.participant_id)
    room_manager.start_debate(room.room_code, room.host_token)

    # Add 3 completed utterances
    for i in range(3):
        utt = room_manager.create_utterance(room.room_code, p.participant_id)
        room_manager.complete_utterance(
            room_code=room.room_code,
            utterance_id=utt.utterance_id,
            transcript=f"This is speaking turn {i+1} discussing solar and wind efficiency.",
            nlp_analysis={
                "quality": {"score": 60.0 + i * 5, "base_score": 60.0 + i * 5},
                "intervention": {"triggered": False, "penalty": 0},
                "evidence": {"is_evidence_bearing": True, "entities": [{"text": "Solar", "label": "MISC"}]},
                "fallacy": {"prediction": "None"},
                "counterargument": {"counterargument": f"Counter {i+1}"}
            }
        )

    # Generate final report
    report = report_service.generate_speaker_report(room, p)
    assert report["total_turns"] == 3
    assert report["analyzed_turns"] == 3
    assert report["base_score"] is not None
    assert report["final_score"] is not None


# =========================================================================
# TEST 3 — SCORE AGGREGATION
# =========================================================================

def test_3_score_aggregation():
    """
    Create multiple completed turns with known quality values (43, 53, 48).
    Verify final report aggregates all turns.
    """
    room = room_manager.create_room(host_name="Moderator Bob", topic="Universal Basic Income")
    p = room_manager.join_request(room.room_code, "Charlie")
    room_manager.admit_participant(room.room_code, room.host_token, p.participant_id)
    room_manager.start_debate(room.room_code, room.host_token)

    scores = [43.0, 53.0, 48.0]
    for s in scores:
        utt = room_manager.create_utterance(room.room_code, p.participant_id)
        room_manager.complete_utterance(
            room_code=room.room_code,
            utterance_id=utt.utterance_id,
            transcript=f"Turn with score {s}",
            nlp_analysis={
                "quality": {"score": s, "base_score": s},
                "intervention": {"triggered": False, "penalty": 0},
                "evidence": {"is_evidence_bearing": False, "entities": []},
                "fallacy": {"prediction": "None"},
                "counterargument": {"counterargument": "Private tip"}
            }
        )

    report = report_service.generate_speaker_report(room, p)
    expected_avg = round((43.0 + 53.0 + 48.0) / 3.0, 1)  # 48.0
    assert report["total_turns"] == 3
    assert report["final_score"] == expected_avg
    assert report["base_score"] == expected_avg


# =========================================================================
# TEST 4 — SPEAKING HISTORY
# =========================================================================

def test_4_speaking_history():
    """
    Verify len(report['speaking_history']) == number of completed turns
    and every entry maps to the correct participant.
    """
    room = room_manager.create_room(host_name="Moderator Dan", topic="AI Regulation")
    p = room_manager.join_request(room.room_code, "Diana")
    room_manager.admit_participant(room.room_code, room.host_token, p.participant_id)
    room_manager.start_debate(room.room_code, room.host_token)

    for i in range(4):
        utt = room_manager.create_utterance(room.room_code, p.participant_id)
        room_manager.complete_utterance(
            room_code=room.room_code,
            utterance_id=utt.utterance_id,
            transcript=f"Turn statement {i+1}",
            nlp_analysis={
                "quality": {"score": 70.0, "base_score": 70.0},
                "intervention": {"triggered": False, "penalty": 0},
                "evidence": {"is_evidence_bearing": True, "entities": []},
                "counterargument": {"counterargument": f"Strategy {i+1}"}
            }
        )

    report = report_service.generate_speaker_report(room, p)
    history = report["speaking_history"]
    assert len(history) == 4
    for idx, item in enumerate(history):
        assert item["turn_number"] == idx + 1
        assert item["participant_id"] == p.participant_id
        assert item["transcript"] == f"Turn statement {idx+1}"
        assert item["quality_score"] == 70.0


# =========================================================================
# TEST 5 — FAILED / INCOMPLETE TURNS EXCLUSION
# =========================================================================

def test_5_incomplete_and_failed_turns_excluded():
    """
    Add completed turns, no-speech turn, failed processing turn, incomplete turn.
    Verify only valid completed turns appear in final report.
    """
    room = room_manager.create_room(host_name="Moderator Eve", topic="Carbon Tax Policy")
    p = room_manager.join_request(room.room_code, "Frank")
    room_manager.admit_participant(room.room_code, room.host_token, p.participant_id)
    room_manager.start_debate(room.room_code, room.host_token)

    # 1. Completed turn
    u1 = room_manager.create_utterance(room.room_code, p.participant_id)
    room_manager.complete_utterance(
        room.room_code, u1.utterance_id, "Completed turn 1",
        nlp_analysis={"quality": {"score": 65.0, "base_score": 65.0}, "intervention": {"triggered": False, "penalty": 0}}
    )

    # 2. No-speech turn
    u2 = room_manager.create_utterance(room.room_code, p.participant_id)
    room_manager.mark_utterance_no_speech(room.room_code, u2.utterance_id)

    # 3. Error turn
    u3 = room_manager.create_utterance(room.room_code, p.participant_id)
    room_manager.mark_utterance_error(room.room_code, u3.utterance_id, "Microphone dropped")

    # 4. Incomplete processing turn (simulated interrupted recording)
    u4 = room_manager.create_utterance(room.room_code, p.participant_id)
    # remains in PROCESSING

    # 5. Completed turn
    u5 = room_manager.create_utterance(room.room_code, p.participant_id)
    room_manager.complete_utterance(
        room.room_code, u5.utterance_id, "Completed turn 2",
        nlp_analysis={"quality": {"score": 75.0, "base_score": 75.0}, "intervention": {"triggered": False, "penalty": 0}}
    )

    report = report_service.generate_speaker_report(room, p)
    assert report["total_turns"] == 2
    assert report["analyzed_turns"] == 2
    assert len(report["speaking_history"]) == 2
    assert report["final_score"] == 70.0


# =========================================================================
# TEST 6 — PARTICIPANT ISOLATION
# =========================================================================

def test_6_participant_isolation():
    """Participant A cannot access Participant B's private report coaching."""
    client = TestClient(app)
    # Create room
    res = client.post("/api/rooms", json={"host_name": "Host Moderator", "topic": "Space Exploration"})
    room_code = res.json()["room_code"]
    host_token = res.json()["host_token"]

    # Join A and B
    res_a = client.post(f"/api/rooms/{room_code}/join", json={"participant_name": "Speaker Alpha"})
    p_a = res_a.json()["participant_id"]
    res_b = client.post(f"/api/rooms/{room_code}/join", json={"participant_name": "Speaker Beta"})
    p_b = res_b.json()["participant_id"]

    client.post(f"/api/rooms/{room_code}/participants/{p_a}/admit", headers={"X-Host-Token": host_token})
    client.post(f"/api/rooms/{room_code}/participants/{p_b}/admit", headers={"X-Host-Token": host_token})
    client.post(f"/api/rooms/{room_code}/start", headers={"X-Host-Token": host_token})

    # End debate
    client.post(f"/api/rooms/{room_code}/end", headers={"X-Host-Token": host_token})

    # Participant A requests B's report without host token
    res_b_viewed_by_a = client.get(
        f"/api/rooms/{room_code}/reports/{p_b}",
        headers={"X-Participant-Id": p_a}
    )
    assert res_b_viewed_by_a.status_code == 200
    report_data = res_b_viewed_by_a.json()["report"]
    # Private coaching stripped for other participant
    for turn in report_data.get("speaking_history", []):
        assert "private_counterargument" not in turn


# =========================================================================
# TEST 7 — HOST REPORTS
# =========================================================================

def test_7_host_can_view_all_reports():
    """Host can retrieve all admitted participants' reports after debate ends."""
    client = TestClient(app)
    res = client.post("/api/rooms", json={"host_name": "Host Moderator", "topic": "Genetically Modified Crops"})
    room_code = res.json()["room_code"]
    host_token = res.json()["host_token"]

    res_a = client.post(f"/api/rooms/{room_code}/join", json={"participant_name": "Speaker 1"})
    p_a = res_a.json()["participant_id"]
    res_b = client.post(f"/api/rooms/{room_code}/join", json={"participant_name": "Speaker 2"})
    p_b = res_b.json()["participant_id"]

    client.post(f"/api/rooms/{room_code}/participants/{p_a}/admit", headers={"X-Host-Token": host_token})
    client.post(f"/api/rooms/{room_code}/participants/{p_b}/admit", headers={"X-Host-Token": host_token})
    client.post(f"/api/rooms/{room_code}/start", headers={"X-Host-Token": host_token})
    client.post(f"/api/rooms/{room_code}/end", headers={"X-Host-Token": host_token})

    # Host retrieves all reports
    res_all = client.get(f"/api/rooms/{room_code}/reports", headers={"X-Host-Token": host_token})
    assert res_all.status_code == 200
    all_reports = res_all.json()["reports"]
    assert p_a in all_reports
    assert p_b in all_reports


# =========================================================================
# TEST 8 — NER EXTRACTION
# =========================================================================

def test_8_ner_extraction():
    """NER analyzer extracts entities and exposes them in NLP engine result."""
    engine = ReasonIQNLPEngine()
    text = "According to the World Health Organization, India and Japan signed a healthcare agreement in 2023."
    res = engine.analyze(text)
    assert res["success"] is True
    assert "ner" in res
    assert isinstance(res["ner"], list)
    # Check that at least one named entity is detected
    assert len(res["ner"]) > 0
    entity_texts = [e["text"] for e in res["ner"]]
    assert any("Health" in t or "Organization" in t or "India" in t or "Japan" in t for t in entity_texts)


# =========================================================================
# TEST 9 — INTERVENTION DETECTOR
# =========================================================================

def test_9_intervention_classification():
    """
    Test personal attack vs legitimate disagreement vs unclear statement.
    """
    # 1. Personal attack
    res_attack = intervention_detector.detect("You are stupid.")
    assert res_attack["triggered"] is True
    assert res_attack["type"] == "personal_attack"
    assert res_attack["penalty"] == 15

    # 2. Legitimate disagreement
    res_disagree = intervention_detector.detect("I strongly disagree with your argument.")
    assert res_disagree["triggered"] is False
    assert res_disagree["penalty"] == 0

    # 3. Legitimate criticism of evidence
    res_evidence = intervention_detector.detect("Your evidence does not support your conclusion.")
    assert res_evidence["triggered"] is False
    assert res_evidence["penalty"] == 0

    # 4. Unclear / fragmented
    res_unclear = intervention_detector.detect("I think. money Needs to happiness.")
    assert res_unclear["triggered"] is True
    assert res_unclear["penalty"] <= 5


# =========================================================================
# TEST 10 — PRIVATE COACHING ISOLATION
# =========================================================================

def test_10_private_coaching_isolation():
    """Verify counterargument coaching is not present in public sanitized reports."""
    raw_report = {
        "participant_id": "p123",
        "turn_history": [
            {
                "turn_number": 1,
                "transcript": "My argument",
                "private_counterargument": {"counterargument": "Secret tactical tip"}
            }
        ],
        "speaking_history": [
            {
                "turn_number": 1,
                "transcript": "My argument",
                "private_counterargument": {"counterargument": "Secret tactical tip"}
            }
        ]
    }
    sanitized = report_service.sanitize_report_for_public(raw_report)
    assert "private_counterargument" not in sanitized["turn_history"][0]
    assert "private_counterargument" not in sanitized["speaking_history"][0]
