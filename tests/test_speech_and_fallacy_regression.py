"""
ReasonIQ - Speech Recognition, Fallacy Detection, Scoring & UI Regression Test Suite

Validates:
1. Audio pipeline: PyAV conversion, empty/corrupt handling, amplitude normalization, isolated temp files.
2. Fallacy regression examples A through L (Bandwagon, False dilemma, Ad hominem,
   Appeal to ignorance, Hasty generalization, False cause, Circular reasoning,
   Straw man, False equivalence, Slippery slope, Negative sentiment without fallacy,
   Insult without substantive argument).
3. No duplicate penalties for ad hominem / personal attack.
4. Non-defaulting dynamic scoring (no hardcoded 65).
5. Explainable coaching and deduction audit trails.
"""

import sys
import io
import wave
import struct
import math
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.nlp.engine import ReasonIQNLPEngine
from backend.services.audio_service import convert_to_wav, process_audio_file, audio_service
from backend.services.room_manager import room_manager
from backend.services.report_service import report_service


@pytest.fixture(scope="module")
def nlp():
    return ReasonIQNLPEngine()


def generate_wav(duration_s=1.0, sample_rate=16000, freq=440.0, silence=False) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        num_frames = int(duration_s * sample_rate)
        frames = bytearray()
        for i in range(num_frames):
            val = 0 if silence else int(32767.0 * 0.5 * math.sin(2.0 * math.pi * freq * (i / sample_rate)))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(frames)
    return buffer.getvalue()


# =========================================================================
# SECTION 1: AUDIO PIPELINE REGRESSION TESTS
# =========================================================================

def test_audio_pipeline_decodes_to_16khz_mono():
    """Verify PyAV correctly normalizes arbitrary audio into 16kHz mono WAV."""
    raw = generate_wav(duration_s=1.0, sample_rate=44100, freq=300.0)
    converted_wav = convert_to_wav(raw, utterance_id="reg_conv_01")
    assert converted_wav is not None
    assert len(converted_wav) > 0
    with wave.open(io.BytesIO(converted_wav), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000


def test_empty_audio_returns_no_speech():
    """Verify 0-byte audio returns NO_SPEECH status."""
    res = process_audio_file(b"", utterance_id="reg_empty_01")
    assert res["status"] == "NO_SPEECH"
    assert res["transcript"] == ""


def test_corrupt_audio_returns_audio_error():
    """Verify unparseable audio container returns AUDIO_ERROR with audio_conversion stage."""
    garbage = b"CORRUPTED_NON_EMPTY_CONTAINER_STREAM_DATA_ABC_123456789"
    res = process_audio_file(garbage, utterance_id="reg_corrupt_01")
    assert res["status"] == "AUDIO_ERROR"
    assert res["stage"] == "audio_conversion"


def test_unique_temporary_files_for_concurrent_utterances():
    """Verify two utterances never collide or share temporary audio files."""
    raw1 = generate_wav(duration_s=0.5, freq=200.0)
    raw2 = generate_wav(duration_s=0.5, freq=400.0)
    p1 = audio_service.convert_to_wav(raw1, utterance_id="client_A_01")
    p2 = audio_service.convert_to_wav(raw2, utterance_id="client_B_01")
    assert p1 != p2
    assert Path(p1).exists()
    assert Path(p2).exists()
    Path(p1).unlink()
    Path(p2).unlink()


# =========================================================================
# SECTION 2: REGRESSION TESTS A THROUGH L (FALLACY DETECTION & CONDUCT)
# =========================================================================

def test_regression_A_bandwagon(nlp):
    """
    A. Bandwagon / appeal to popularity
    Input: 'Everyone uses AI art, so AI art must be better than traditional art.'
    Expected: Ad populum / Bandwagon signal with confidence value and explanation
              that popularity does not establish superiority.
    """
    text = "Everyone uses AI art, so AI art must be better than traditional art."
    res = nlp.analyze(text)
    assert res["success"] is True
    assert res["fallacy"] is not None
    assert res["fallacy"]["prediction"] in ["ad populum", "bandwagon"]
    assert "bandwagon" in res["fallacy"]["label"].lower() or "ad populum" in res["fallacy"]["label"].lower()
    assert res["fallacy"]["decision_strength"] >= 0.45
    assert "popularity" in res["fallacy"]["explanation"].lower() or "widespread" in res["fallacy"]["explanation"].lower()


def test_regression_B_false_dilemma(nlp):
    """
    B. False dilemma
    Input: 'Either we support traditional art or we completely destroy our culture.'
    Expected: False dilemma signal presenting two extreme alternatives.
    """
    text = "Either we support traditional art or we completely destroy our culture."
    res = nlp.analyze(text)
    assert res["success"] is True
    assert res["fallacy"] is not None
    assert res["fallacy"]["prediction"] == "false dilemma"
    assert "extreme" in res["fallacy"]["explanation"].lower() or "alternatives" in res["fallacy"]["explanation"].lower()


def test_regression_C_ad_hominem(nlp):
    """
    C. Ad hominem
    Input: 'You cannot trust his opinion on AI because he is just a fool.'
    Expected: Detect personal attack against the person as an Ad hominem signal.
              Also trigger relevant personal-attack intervention.
    """
    text = "You cannot trust his opinion on AI because he is just a fool."
    res = nlp.analyze(text)
    assert res["success"] is True
    assert res["fallacy"] is not None
    assert res["fallacy"]["prediction"] == "ad hominem"
    # Also triggers personal attack intervention
    assert res["intervention"]["triggered"] is True
    assert res["intervention"]["type"] == "personal_attack"


def test_regression_D_appeal_to_ignorance(nlp):
    """
    D. Appeal to ignorance
    Input: 'Nobody has proved that AI cannot create better art, so AI is definitely superior.'
    Expected: Appeal to ignorance signal because absence of disproof is treated as proof.
    """
    text = "Nobody has proved that AI cannot create better art, so AI is definitely superior."
    res = nlp.analyze(text)
    assert res["success"] is True
    assert res["fallacy"] is not None
    assert res["fallacy"]["prediction"] == "appeal to ignorance"
    assert "disproof" in res["fallacy"]["explanation"].lower() or "proof" in res["fallacy"]["explanation"].lower()


def test_regression_E_hasty_generalization(nlp):
    """
    E. Hasty generalization
    Input: 'One student cheated using AI, so all students who use AI are cheaters.'
    Expected: Hasty generalization signal.
    """
    text = "One student cheated using AI, so all students who use AI are cheaters."
    res = nlp.analyze(text)
    assert res["success"] is True
    assert res["fallacy"] is not None
    assert res["fallacy"]["prediction"] in ["faulty generalization", "hasty generalization"]
    assert "generalization" in res["fallacy"]["label"].lower()


def test_regression_F_false_cause(nlp):
    """
    F. False cause
    Input: 'Students started using AI and their marks dropped, so AI must have caused the drop.'
    Expected: False cause signal inferring causation from association without evidence.
    """
    text = "Students started using AI and their marks dropped, so AI must have caused the drop."
    res = nlp.analyze(text)
    assert res["success"] is True
    assert res["fallacy"] is not None
    assert res["fallacy"]["prediction"] in ["false causality", "false cause"]
    assert "caus" in res["fallacy"]["explanation"].lower()


def test_regression_G_circular_reasoning(nlp):
    """
    G. Circular reasoning
    Input: 'AI art is better because it is superior to traditional art.'
    Expected: Circular reasoning signal restating conclusion as support.
    """
    text = "AI art is better because it is superior to traditional art."
    res = nlp.analyze(text)
    assert res["success"] is True
    assert res["fallacy"] is not None
    assert res["fallacy"]["prediction"] == "circular reasoning"
    assert "conclusion" in res["fallacy"]["explanation"].lower() or "premise" in res["fallacy"]["explanation"].lower()


def test_regression_H_straw_man(nlp):
    """
    H. Straw man
    Input: 'You want to ban all technology from every school.'
    Expected: Straw man signal misrepresenting position into an extreme claim.
    """
    text = "You want to ban all technology from every school."
    res = nlp.analyze(text)
    assert res["success"] is True
    assert res["fallacy"] is not None
    assert res["fallacy"]["prediction"] in ["fallacy of extension", "straw man"]
    assert "straw man" in res["fallacy"]["label"].lower() or "extension" in res["fallacy"]["label"].lower()


def test_regression_I_false_equivalence(nlp):
    """
    I. False equivalence
    Input: 'Using an AI spelling assistant is exactly the same as submitting a completely AI-generated assignment.'
    Expected: False equivalence signal treating materially different activities as identical.
    """
    text = "Using an AI spelling assistant is exactly the same as submitting a completely AI-generated assignment."
    res = nlp.analyze(text)
    assert res["success"] is True
    assert res["fallacy"] is not None
    assert res["fallacy"]["prediction"] in ["equivocation", "false equivalence", "fallacy of relevance"]
    assert "equivalence" in res["fallacy"]["label"].lower() or "equivocation" in res["fallacy"]["label"].lower()


def test_regression_J_slippery_slope(nlp):
    """
    J. Slippery slope
    Input: 'If one student is allowed to use AI, soon every student will stop thinking, nobody will study, and schools will become useless.'
    Expected: Slippery slope signal predicting escalating catastrophic chain.
    """
    text = "If one student is allowed to use AI, soon every student will stop thinking, nobody will study, and schools will become useless."
    res = nlp.analyze(text)
    assert res["success"] is True
    assert res["fallacy"] is not None
    assert res["fallacy"]["prediction"] == "slippery slope"
    assert "chain" in res["fallacy"]["explanation"].lower() or "consequences" in res["fallacy"]["explanation"].lower()


def test_regression_K_negative_sentiment_without_fallacy(nlp):
    """
    K. Negative sentiment without a fallacy
    Input: 'I strongly disagree with that proposal because the evidence is insufficient.'
    Expected: Negative/disagreement sentiment, but NO fallacy penalty, NO personal attack.
    """
    text = "I strongly disagree with that proposal because the evidence is insufficient."
    res = nlp.analyze(text)
    assert res["success"] is True
    # Negative sentiment is acknowledged
    assert res["sentiment"]["label"] == "negative"
    # But NO fallacy is flagged
    assert res["fallacy"] is None
    # And NO conduct penalty is applied
    assert res["intervention"]["triggered"] is False
    assert res["quality"]["intervention_penalty"] == 0.0
    # Clean quality score reflecting structured argument
    assert res["quality"]["score"] >= 60.0


def test_regression_L_insult_without_substantive_argument(nlp):
    """
    L. Insult without a substantive argument
    Input: 'You are stupid.'
    Expected: Personal-attack intervention (-15 points). Do NOT invent a reasoning fallacy.
    """
    text = "You are stupid."
    res = nlp.analyze(text)
    assert res["success"] is True
    assert res["intervention"]["triggered"] is True
    assert res["intervention"]["type"] == "personal_attack"
    assert res["intervention"]["penalty"] == 15
    # No reasoning fallacy invented for a bare insult
    assert res["fallacy"] is None
    # Score properly reflects penalty
    assert res["quality"]["score"] <= 40.0


# =========================================================================
# SECTION 3: SCORING & DEDUCTION AUDIT INTEGRITY
# =========================================================================

def test_no_hardcoded_65_default(nlp):
    """Verify different debate statements yield distinct dynamic quality scores, not 65.0."""
    statements = [
        "AI provides personalized learning.",
        "According to UNESCO 2023 statistics, 45% of classrooms utilize interactive digital tools for learning.",
        "I believe that traditional art embodies human creativity and emotion.",
        "Yes."
    ]
    scores = set()
    for s in statements:
        res = nlp.analyze(s)
        if res.get("quality", {}).get("score") is not None:
            scores.add(res["quality"]["score"])
    # Scores must vary based on evidence, length, specificity
    assert len(scores) >= 2


def test_no_double_deduction_for_ad_hominem(nlp):
    """Verify that an ad hominem statement does NOT deduct 30 points (15 from fallacy + 15 from conduct)."""
    text = "You cannot trust his opinion on AI because he is just a fool."
    res = nlp.analyze(text)
    quality = res["quality"]
    # Base score should be in reasonable 50-70 range
    base = quality["base_score"]
    final = quality["score"]
    # Deduction should be exactly 15 points (the personal attack penalty), NOT 30
    assert round(base - final, 1) == 15.0


def test_social_statement_has_no_artificial_score(nlp):
    """Verify 'Thank you.' is recognized as social statement with score=None, avoiding skewing debate averages."""
    res = nlp.analyze("Thank you.")
    assert res["content_type"] == "social_statement"
    assert res["quality"]["score"] is None
    assert res["quality"]["level"] == "Not Applicable"


# =========================================================================
# SECTION 4: LIVE FAILURE CASES (SECTION C & CONVERSATIONAL CONTEXT D)
# =========================================================================

def test_section_c_understandable_position_with_acronym(nlp):
    """
    Case C1: 'I agree A.I. should support even creativity not replace it.'
    Must NOT be labeled as fragmented or unclear; it is an understandable position.
    """
    text = "I agree A.I. should support even creativity not replace it."
    res = nlp.analyze(text)
    assert res["success"] is True
    assert res["content_type"] == "substantive_argument"
    assert res["quality"]["clarity_status"] in ["Mostly Clear", "Clear & Structured", "Needs Clarification"]
    assert res["quality"]["clarity_status"] != "Fragmented or Unclear"
    assert res["quality"]["score"] is not None


def test_section_c_personal_defiance_retort(nlp):
    """
    Case C2: 'You don't tell me what to do. I will do what I want.'
    Must NOT be called unintelligible; classified as non-substantive / insufficient argument.
    """
    text = "You don't tell me what to do. I will do what I want."
    res = nlp.analyze(text)
    assert res["success"] is True
    assert res["content_type"] == "insufficient_argument"
    assert res["quality"]["clarity_status"] != "Fragmented or Unclear"


def test_section_c_anecdotal_painting_hasty_generalization(nlp):
    """
    Case C3: 'My friend used AI to create an amazing painting in seconds. Therefore AI art is better than all traditional art.'
    Reasoning is understandable; detects hasty generalization / anecdotal generalization.
    """
    text = "My friend used AI to create an amazing painting in seconds. Therefore AI art is better than all traditional art."
    res = nlp.analyze(text)
    assert res["success"] is True
    assert res["fallacy"] is not None
    assert res["fallacy"]["prediction"] == "faulty generalization"
    assert "Hasty generalization" in res["fallacy"]["label"]


def test_section_c_grandfather_hasty_generalization(nlp):
    """
    Case C4: 'My grandfather never used computers and was successful, so computers are useless.'
    Generalization from one single example to an entire category.
    """
    text = "My grandfather never used computers and was successful, so computers are useless."
    res = nlp.analyze(text)
    assert res["success"] is True
    assert res["fallacy"] is not None
    assert res["fallacy"]["prediction"] == "faulty generalization"
    assert "Hasty generalization" in res["fallacy"]["label"]


def test_section_c_false_cause_with_turn_starter(nlp):
    """
    Case C5: 'I will. College introduced AI tools, students' marks decreased. Therefore, AI tools caused the decline.'
    Assesses false-cause fallacy from temporal association.
    """
    text = "I will. College introduced AI tools, students' marks decreased. Therefore, AI tools caused the decline."
    res = nlp.analyze(text)
    assert res["success"] is True
    assert res["fallacy"] is not None
    assert res["fallacy"]["prediction"] == "false causality"
    assert "False cause" in res["fallacy"]["label"]


def test_section_d_slippery_slope_artists_disappear(nlp):
    """
    Section D Slippery Slope: 'If we allow AI to create paintings, soon all human artists will lose their jobs and disappear forever.'
    """
    text = "If we allow AI to create paintings, soon all human artists will lose their jobs and disappear forever."
    res = nlp.analyze(text)
    assert res["success"] is True
    assert res["fallacy"] is not None
    assert res["fallacy"]["prediction"] == "slippery slope"
    assert "Slippery slope" in res["fallacy"]["label"]


def test_section_d_appeal_to_authority_expert(nlp):
    """
    Section D Appeal to Authority: 'An expert supports this claim, so it must be true and needs no other evidence.'
    """
    text = "An expert supports this claim, so it must be true and needs no other evidence."
    res = nlp.analyze(text)
    assert res["success"] is True
    assert res["fallacy"] is not None
    assert res["fallacy"]["prediction"] == "fallacy of credibility"
    assert "Appeal to authority" in res["fallacy"]["label"]


def test_section_d_strawman_with_and_without_context(nlp):
    """
    Section D Straw Man:
    Speaker A: 'AI should be limited in some classroom assessments.'
    Speaker B: 'You want to ban all technology in every school.'
    When preceded by qualified turn, high-confidence Straw Man signal.
    When isolated, cautious signal.
    """
    speaker_a = "AI should be limited in some classroom assessments."
    speaker_b = "You want to ban all technology in every school."

    # With preceding turn context
    res_with_ctx = nlp.analyze(speaker_b, context_prev_text=speaker_a)
    assert res_with_ctx["fallacy"] is not None
    assert res_with_ctx["fallacy"]["prediction"] == "fallacy of extension"
    assert res_with_ctx["fallacy"]["decision_strength"] >= 0.75

    # Without context (isolated sentence) -> Cautious result
    res_isolated = nlp.analyze(speaker_b, context_prev_text=None)
    assert res_isolated["fallacy"] is not None
    assert res_isolated["fallacy"]["decision_strength"] <= 0.70
    assert "Possible" in res_isolated["fallacy"]["label"] or "Straw man" in res_isolated["fallacy"]["label"]

