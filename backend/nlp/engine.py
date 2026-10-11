import time
from typing import Dict, Any, List, Optional

from .argument_quality import ArgumentQualityAnalyzer
from .counterargument import CounterargumentGenerator
from .intervention import intervention_detector
from .content_gate import content_gate


class ReasonIQNLPEngine:

    def __init__(self):

        print("=" * 80)
        print("Initializing ReasonIQ NLP Engine...")
        print("=" * 80)

        start_time = time.time()

        self.gate = content_gate
        self.argument_quality = ArgumentQualityAnalyzer()
        self.counterargument = CounterargumentGenerator()
        self.intervention_detector = intervention_detector

        elapsed = time.time() - start_time

        print("=" * 80)
        print(
            f"ReasonIQ NLP Engine ready "
            f"({elapsed:.2f} seconds)"
        )
        print("=" * 80)

    # =========================================================
    # GENERIC VALUE EXTRACTION
    # =========================================================

    @staticmethod
    def first_value(data, keys, default=None):

        if not isinstance(data, dict):
            return default

        for key in keys:

            if key in data and data[key] is not None:
                return data[key]

        return default

    # =========================================================
    # FIND ANALYSIS DICTIONARY
    # =========================================================

    @staticmethod
    def find_analysis_dict(result, names):

        if not isinstance(result, dict):
            return {}

        # Direct lookup
        for name in names:

            value = result.get(name)

            if isinstance(value, dict):
                return value

        # Case-insensitive lookup
        lower_map = {
            str(key).lower(): key
            for key in result.keys()
        }

        for name in names:

            actual_key = lower_map.get(
                name.lower()
            )

            if actual_key is not None:

                value = result.get(
                    actual_key
                )

                if isinstance(value, dict):
                    return value

        return {}

    # =========================================================
    # FLOAT CONVERSION
    # =========================================================

    @staticmethod
    def to_float(value, default=None):

        if value is None:
            return default

        try:
            return round(
                float(value),
                4
            )

        except (
            TypeError,
            ValueError
        ):
            return default

    # =========================================================
    # NORMALIZE FALLACY (Suppress 'intentional' & non-fallacies)
    # =========================================================

    def normalize_fallacy(self, raw_fallacy: Dict[str, Any], text: str) -> Optional[Dict[str, Any]]:
        if not raw_fallacy or not isinstance(raw_fallacy, dict):
            return None

        prediction = raw_fallacy.get("prediction")
        if not prediction:
            return None

        pred_str = str(prediction).strip().lower()

        # Suppress dataset artifact labels
        if pred_str in ["none", "unknown", "intentional", "miscellaneous", ""]:
            return None

        reliable = raw_fallacy.get("reliable", True)
        if not reliable:
            return None

        strength = self.to_float(raw_fallacy.get("decision_strength"), 0.0)
        display_label = raw_fallacy.get("label") or pred_str.capitalize()
        explanation = raw_fallacy.get("explanation", "")

        return {
            "prediction": pred_str,
            "label": display_label,
            "decision_strength": strength,
            "reliable": True,
            "explanation": explanation
        }

    # =========================================================
    # MAIN ANALYSIS PIPELINE (Gated)
    # =========================================================

    def analyze(
        self,
        text: str,
        transcription_warning: Optional[str] = None,
        context_prev_text: Optional[str] = None
    ) -> Dict[str, Any]:

        if not text or not text.strip():

            return {
                "success": False,
                "text": text,
                "error": "Empty text provided."
            }

        clean_text = text.strip()

        # =====================================================
        # STAGE 1: SPEECH / CONTENT QUALITY GATE
        # =====================================================
        gate_res = self.gate.classify(clean_text, transcription_warning=transcription_warning)
        content_type = gate_res.get("content_type", "substantive_argument")

        # -----------------------------------------------------
        # CASE A: SOCIAL STATEMENT / GREETING (Non-argument)
        # -----------------------------------------------------
        if content_type == "social_statement":
            sentiment = self.argument_quality.sentiment_analyzer.analyze(clean_text)

            return {
                "success": True,
                "text": clean_text,
                "content_type": "social_statement",
                "argument_component": None,
                "fallacy": None,
                "bias": {
                    "prediction": "0",
                    "label": "Not detected",
                    "status": "not_detected",
                    "decision_strength": 0.0
                },
                "sentiment": sentiment,
                "evidence": {
                    "status": "not_applicable",
                    "is_evidence_bearing": False,
                    "evidence_presence_score": 0.0,
                    "sources": [],
                    "statistics": [],
                    "dates": [],
                    "indicators": [],
                    "entities": []
                },
                "ner": [],
                "intervention": {
                    "triggered": False,
                    "type": None,
                    "severity": "none",
                    "warning": "",
                    "penalty": 0,
                    "details": "Social statement / non-argument."
                },
                "quality": {
                    "score": None,
                    "base_score": None,
                    "level": "Not Applicable",
                    "intervention_penalty": 0,
                    "specificity_score": 0.0,
                    "clarity_status": "Non-substantive",
                    "clarity_score": None,
                    "explanation": [
                        "Social pleasantry or greeting — no substantive debate argument to evaluate."
                    ]
                },
                "counterargument": None
            }

        # -----------------------------------------------------
        # CASE B: UNCLEAR / FRAGMENTED / ASR-CORRUPTED
        # -----------------------------------------------------
        if content_type == "unclear_fragment":
            sentiment = self.argument_quality.sentiment_analyzer.analyze(clean_text)
            intervention = self.intervention_detector.detect(clean_text)
            counterarg = self.counterargument.generate(
                text=clean_text,
                intervention_analysis=intervention
            )

            return {
                "success": True,
                "text": clean_text,
                "content_type": "unclear_fragment",
                "argument_component": {
                    "prediction": "Unclear",
                    "decision_strength": 0.0
                },
                "fallacy": None,
                "bias": {
                    "prediction": "0",
                    "label": "Not detected",
                    "status": "not_detected",
                    "decision_strength": 0.0
                },
                "sentiment": sentiment,
                "evidence": {
                    "status": "limited",
                    "is_evidence_bearing": False,
                    "evidence_presence_score": 0.0,
                    "sources": [],
                    "statistics": [],
                    "dates": [],
                    "indicators": [],
                    "entities": []
                },
                "ner": [],
                "intervention": intervention,
                "quality": {
                    "score": 45.0,
                    "base_score": 50.0,
                    "level": "Needs Review",
                    "intervention_penalty": 5.0,
                    "specificity_score": 0.0,
                    "clarity_status": "Fragmented / Unclear",
                    "clarity_score": 45.0,
                    "explanation": [
                        "Statement is unclear or fragmented; syntax and semantic completeness need clarification.",
                        "Intervention penalty applied: -5 points (Unclear statement)."
                    ]
                },
                "counterargument": counterarg
            }

        # -----------------------------------------------------
        # CASE C: INSUFFICIENT ARGUMENT (Bare assertion)
        # -----------------------------------------------------
        if content_type == "insufficient_argument":
            sentiment = self.argument_quality.sentiment_analyzer.analyze(clean_text)
            intervention = self.intervention_detector.detect(clean_text)
            counterarg = self.counterargument.generate(
                text=clean_text,
                intervention_analysis=intervention
            )

            return {
                "success": True,
                "text": clean_text,
                "content_type": "insufficient_argument",
                "argument_component": {
                    "prediction": "Claim",
                    "decision_strength": 0.50
                },
                "fallacy": None,
                "bias": {
                    "prediction": "0",
                    "label": "Not detected",
                    "status": "not_detected",
                    "decision_strength": 0.0
                },
                "sentiment": sentiment,
                "evidence": {
                    "status": "limited",
                    "is_evidence_bearing": False,
                    "evidence_presence_score": 0.0,
                    "sources": [],
                    "statistics": [],
                    "dates": [],
                    "indicators": [],
                    "entities": []
                },
                "ner": [],
                "intervention": intervention,
                "quality": {
                    "score": 47.0,
                    "base_score": 50.0,
                    "level": "Needs Review",
                    "intervention_penalty": 3.0,
                    "specificity_score": 0.0,
                    "clarity_status": "Needs Clarification",
                    "clarity_score": 60.0,
                    "explanation": [
                        "The statement identifies an assertion but does not provide supporting reasoning or explanation.",
                        "Intervention penalty applied: -3 points (Insufficient argument)."
                    ]
                },
                "counterargument": counterarg
            }

        # =====================================================
        # STAGE 2: SUBSTANTIVE ARGUMENTS & PERSONAL ATTACKS
        # =====================================================
        raw_result = self.argument_quality.analyze(clean_text, context_prev_text=context_prev_text)
        intervention = self.intervention_detector.detect(clean_text)

        # Fallacy normalization (suppresses 'intentional' and unverified fallacies)
        fallacy = self.normalize_fallacy(raw_result.get("fallacy", {}), clean_text)

        # Gate fallacy if audio transcription is warned/unreliable or utterance is personal attack
        if transcription_warning:
            fallacy = None

        if content_type == "personal_attack":
            # If text has ad hominem structure, retain ad hominem; otherwise fallacy is None
            if fallacy and fallacy.get("prediction") != "ad hominem":
                fallacy = None

        bias = raw_result.get("bias", {})
        sentiment = raw_result.get("sentiment", {})
        evidence = raw_result.get("evidence", {})
        argument_component = raw_result.get("argument_component", {})

        # Calculate final quality score
        base_score = raw_result.get("base_quality_score", 50.0)
        penalty = float(intervention.get("penalty", 0)) if intervention.get("triggered") else 0.0
        fallacy_pen = float(raw_result.get("fallacy_penalty", 0.0))
        bias_pen = float(raw_result.get("bias_penalty", 0.0))
        final_score = max(0.0, min(100.0, float(base_score) - penalty - fallacy_pen - bias_pen))

        if final_score >= 75.0:
            level = "Strong"
        elif final_score >= 50.0:
            level = "Moderate"
        else:
            level = "Needs Review"

        quality_explanations = list(raw_result.get("explanation", []))
        if penalty > 0:
            quality_explanations.append(
                f"Intervention penalty applied: -{int(penalty)} points ({intervention.get('warning')})"
            )

        quality = {
            "score": self.to_float(final_score, 50.0),
            "base_score": self.to_float(base_score, 50.0),
            "intervention_penalty": self.to_float(penalty, 0.0),
            "fallacy_penalty": self.to_float(fallacy_pen, 0.0),
            "bias_penalty": self.to_float(bias_pen, 0.0),
            "level": level,
            "specificity_score": raw_result.get("specificity_score", 0.0),
            "clarity_status": gate_res.get("clarity_status", "Clear & Structured"),
            "clarity_score": gate_res.get("clarity_score", 90.0),
            "explanation": quality_explanations
        }

        # Counterargument generation
        counterargument = self.counterargument.generate(
            text=clean_text,
            fallacy_analysis=fallacy,
            bias_analysis=bias,
            evidence_analysis=evidence,
            quality_analysis={"specificity_score": quality["specificity_score"]},
            intervention_analysis=intervention
        )

        return {
            "success": True,
            "text": clean_text,
            "content_type": content_type,
            "argument_component": argument_component,
            "fallacy": fallacy,
            "bias": bias,
            "sentiment": sentiment,
            "evidence": evidence,
            "ner": evidence.get("entities", []),
            "intervention": intervention,
            "quality": quality,
            "counterargument": counterargument
        }

    # =========================================================
    # BATCH ANALYSIS
    # =========================================================

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:

        if not texts:
            return []

        results = []

        for text in texts:
            results.append(self.analyze(text))

        return results


# Backward compatibility alias
BiasLessNLPEngine = ReasonIQNLPEngine