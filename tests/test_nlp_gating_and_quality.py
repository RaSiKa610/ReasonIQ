"""
ReasonIQ - Speech & Content Gate and NLP Gating Automated Test Suite

Validates the 16 physical debate test requirements including:
- Social statement handling without fallacy or counterargument forcing
- Conservative fallacy gating (no forcing on normal arguments)
- Genuine fallacy detection preservation (false dilemma, faulty generalization, ad populum, ad hominem)
- Personal attack conduct interventions with ad hominem separation
- Suppression of dataset artifacts like 'intentional' from logical fallacies
- Clean NER extraction without generic 'AI' or ASR fragments
- Conservative bias terminology and thresholds
- Explainable argument quality scores
"""

import pytest
from backend.nlp.engine import ReasonIQNLPEngine


@pytest.fixture(scope="module")
def nlp_engine():
    return ReasonIQNLPEngine()


def test_1_social_statement_thank_you(nlp_engine):
    """TEST 1: 'Thank you.' is handled as a social statement without fake premise/fallacy/counterarg."""
    res = nlp_engine.analyze("Thank you.")
    assert res["success"] is True
    assert res["content_type"] == "social_statement"
    assert res["argument_component"] is None
    assert res["fallacy"] is None
    assert res["counterargument"] is None
    assert res["evidence"]["status"] == "not_applicable"
    assert res["quality"]["score"] is None


def test_2_greeting_good_morning(nlp_engine):
    """TEST 2: 'Good morning everyone.' is non-argument without forced fallacy."""
    res = nlp_engine.analyze("Good morning everyone.")
    assert res["success"] is True
    assert res["content_type"] == "social_statement"
    assert res["fallacy"] is None
    assert res["counterargument"] is None


def test_3_legitimate_argument_no_forced_fallacy(nlp_engine):
    """TEST 3: Normal substantive argument does not get forced into a fallacy."""
    res = nlp_engine.analyze("AI can help students understand difficult topics by providing personalized explanations.")
    assert res["success"] is True
    assert res["content_type"] == "substantive_argument"
    assert res["fallacy"] is None  # no forced fallacy
    assert res["quality"]["score"] >= 50.0


def test_4_strong_false_dilemma(nlp_engine):
    """TEST 4: Strong false dilemma is correctly detected."""
    res = nlp_engine.analyze("Either we allow AI in school or students will never be ready for the future.")
    assert res["success"] is True
    assert res["fallacy"] is not None
    assert res["fallacy"]["prediction"] == "false dilemma"
    assert "false dilemma" in res["counterargument"]["reason"].lower()


def test_5_strong_faulty_generalization(nlp_engine):
    """TEST 5: Strong faulty generalization is correctly detected."""
    res = nlp_engine.analyze("One student used AI to cheat, so all students will use AI to cheat.")
    assert res["success"] is True
    assert res["fallacy"] is not None
    assert res["fallacy"]["prediction"] == "faulty generalization"
    assert "faulty generalization" in res["counterargument"]["reason"].lower()


def test_6_ad_populum(nlp_engine):
    """TEST 6: Ad populum is correctly detected."""
    res = nlp_engine.analyze("Millions of students use it, so it must be the best way to learn.")
    assert res["success"] is True
    assert res["fallacy"] is not None
    assert res["fallacy"]["prediction"] == "ad populum"
    assert "ad populum" in res["counterargument"]["reason"].lower()


def test_7_ad_hominem_and_personal_attack(nlp_engine):
    """TEST 7: Ad hominem with attack on speaker triggers fallacy and personal attack intervention."""
    res = nlp_engine.analyze("You don't understand AI, so your opinion against it doesn't matter.")
    assert res["success"] is True
    assert res["fallacy"] is not None
    assert res["fallacy"]["prediction"] == "ad hominem"
    assert res["intervention"]["triggered"] is True
    assert res["intervention"]["type"] == "personal_attack"
    assert res["intervention"]["penalty"] == 15


def test_8_personal_attack_no_intentional_fallacy(nlp_engine):
    """TEST 8: Personal attack triggers penalty -15 and suppresses 'intentional' from logical fallacy."""
    res = nlp_engine.analyze("You are stupid.")
    assert res["success"] is True
    assert res["intervention"]["triggered"] is True
    assert res["intervention"]["type"] == "personal_attack"
    assert res["intervention"]["penalty"] == 15
    # Logical fallacy must NOT expose 'intentional'
    if res["fallacy"] is not None:
        assert res["fallacy"].get("prediction") != "intentional"


def test_9_legitimate_disagreement_no_personal_attack(nlp_engine):
    """TEST 9: Legitimate debate disagreement does NOT trigger personal attack."""
    res = nlp_engine.analyze("I strongly disagree with your argument.")
    assert res["success"] is True
    assert res["intervention"]["triggered"] is False


def test_10_evidence_weakness_not_forced_fallacy(nlp_engine):
    """TEST 10: Broad claim with weak evidence does NOT get forced into faulty generalization."""
    res = nlp_engine.analyze("AI is useful for students.")
    assert res["success"] is True
    assert res["fallacy"] is None
    assert res["evidence"]["is_evidence_bearing"] is False
    assert "evidence" in res["counterargument"]["strategy"].lower() or "reasoning" in res["counterargument"]["strategy"].lower()


def test_11_low_information_statement(nlp_engine):
    """TEST 11: Low-information assertion is classified as insufficient argument without forced fallacy."""
    res = nlp_engine.analyze("There is many problems with it.")
    assert res["success"] is True
    assert res["content_type"] in ["insufficient_argument", "low_information"]
    assert res["fallacy"] is None
    assert res["intervention"]["triggered"] is True


def test_12_unclear_transcript_asr_artifacts(nlp_engine):
    """TEST 12: ASR corrupted statement triggers unclear warning and filters out bogus person 'Gan'."""
    res = nlp_engine.analyze("students Ganlones faster with AI.")
    assert res["success"] is True
    assert res["content_type"] == "unclear_fragment"
    assert res["intervention"]["triggered"] is True
    assert res["intervention"]["type"] == "unclear_statement"
    # Ensure 'Gan' is NOT extracted as a person
    entity_texts = [e["text"] for e in res["ner"]]
    assert "Gan" not in entity_texts


def test_13_ner_generic_ai_filtered(nlp_engine):
    """TEST 13: Standalone generic 'AI' is filtered from Named Entities."""
    res = nlp_engine.analyze("AI can help students.")
    assert res["success"] is True
    entity_texts = [e["text"] for e in res["ner"]]
    assert "AI" not in entity_texts


def test_14_real_ner_preserved(nlp_engine):
    """TEST 14: Legitimate named entities (OpenAI, Mumbai) are preserved."""
    res = nlp_engine.analyze("OpenAI researchers presented the system in Mumbai.")
    assert res["success"] is True
    entity_texts = [e["text"] for e in res["ner"]]
    assert "OpenAI" in entity_texts
    assert "Mumbai" in entity_texts


def test_15_bias_conservative_behavior(nlp_engine):
    """TEST 15: Ordinary debate statements are not branded as 'Detected'."""
    res = nlp_engine.analyze("AI should be used in education.")
    assert res["success"] is True
    assert res["bias"]["status"] != "detected"


def test_16_counterargument_gating(nlp_engine):
    """TEST 16: Counterarguments respect content gate (none for social, constructive for insufficient)."""
    res_social = nlp_engine.analyze("Thank you.")
    assert res_social["counterargument"] is None

    res_insufficient = nlp_engine.analyze("There is many problems with it.")
    assert res_insufficient["counterargument"] is not None
    assert "reasoning" in res_insufficient["counterargument"]["strategy"] or "expansion" in res_insufficient["counterargument"]["strategy"]
