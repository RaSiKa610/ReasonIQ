import os
import re
import joblib
from typing import Dict, Any, List, Optional, Tuple

from .evidence import EvidenceExtractor
from .sentiment import SentimentAnalyzer


class ArgumentQualityAnalyzer:

    def __init__(self):

        print("Initializing Argument Quality Analyzer...")

        # =====================================================
        # Evidence Analyzer
        # =====================================================

        self.evidence_analyzer = EvidenceExtractor()

        # =====================================================
        # Sentiment Analyzer
        # =====================================================

        self.sentiment_analyzer = SentimentAnalyzer()

        # =====================================================
        # Model Directory
        # =====================================================

        base_dir = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            )
        )

        self.models_dir = os.path.join(
            base_dir,
            "models"
        )

        # =====================================================
        # Load AAEC Argument Component Model
        # =====================================================

        self.aaec_model = self._load_model(
            "aaec_tfidf_svm_improved.pkl"
        )

        # =====================================================
        # Load LOGIC Fallacy Model
        # =====================================================

        self.logic_model = self._load_model(
            "logic_tfidf_svm_improved.pkl"
        )

        # =====================================================
        # Load WNC Bias-Aware Model
        # =====================================================

        self.wnc_model = self._load_model(
            "wnc_tfidf_svm_improved.pkl"
        )

        # Configurable thresholds
        self.fallacy_display_threshold = 0.65
        self.bias_display_threshold = 0.65

        print("Argument Quality Analyzer ready.")

    # =========================================================
    # MODEL LOADING
    # =========================================================

    def _load_model(self, filename):

        path = os.path.join(
            self.models_dir,
            filename
        )

        if not os.path.exists(path):

            print(
                f"Warning: Model not found: {path}"
            )

            return None

        try:

            model = joblib.load(path)

            print(
                f"Loaded model: {filename}"
            )

            return model

        except Exception as error:

            print(
                f"Could not load {filename}: {error}"
            )

            return None

    # =========================================================
    # CONVERT SVM DECISION TO DISPLAY STRENGTH
    # =========================================================

    def _decision_strength(
        self,
        decision
    ):

        """
        Converts an SVM decision margin into a bounded
        0-1 strength value for internal scoring.

        This is NOT a probability.
        """

        value = abs(
            float(decision)
        )

        strength = (
            value / (1.0 + value)
        )

        return round(
            strength,
            4
        )

    # =========================================================
    # ARGUMENT COMPONENT CLASSIFICATION
    # =========================================================

    def predict_argument_component(
        self,
        text
    ):

        if self.aaec_model is None:

            return {
                "label": "unknown",
                "decision_strength": 0.0
            }

        try:

            prediction = self.aaec_model.predict(
                [text]
            )[0]

            decision_strength = 0.0

            if hasattr(
                self.aaec_model,
                "decision_function"
            ):

                decision = (
                    self.aaec_model.decision_function(
                        [text]
                    )
                )

                if len(decision.shape) == 1:

                    decision_strength = (
                        self._decision_strength(
                            decision[0]
                        )
                    )

                else:

                    maximum_margin = max(
                        abs(float(value))
                        for value in decision[0]
                    )

                    decision_strength = (
                        self._decision_strength(
                            maximum_margin
                        )
                    )

            return {
                "label": str(
                    prediction
                ),
                "decision_strength":
                    decision_strength
            }

        except Exception as error:

            print(
                "Argument component prediction error:",
                error
            )

            return {
                "label": "unknown",
                "decision_strength": 0.0
            }

    # =========================================================
    # FALLACY DETECTION (With Structural & Reliability Checks)
    # =========================================================

    def is_reliable_fallacy(
        self,
        fallacy_label: Optional[str],
        decision_strength: float,
        text: str
    ) -> bool:
        """
        Conservative fallacy verification. Prevents forcing fallacies
        on normal claims, greetings, or low-confidence margins.
        """
        if not fallacy_label:
            return False

        label = str(fallacy_label).strip().lower()

        # Suppress non-fallacy dataset artifacts
        if label in ["", "none", "unknown", "intentional", "miscellaneous"]:
            return False

        text_lower = text.lower()

        # Structural linguistic checks for high-impact fallacies
        if label == "false dilemma":
            return bool(re.search(r"\b(either\b.*\bor\b|if\s+(we\s+)?(don't|do\s+not)\b.*(never|no\s+future|fall\s+behind)|only\s+two\s+(options|choices)|must\s+choose\s+between)\b", text_lower))

        if label == "faulty generalization":
            return bool(re.search(r"\b(one\s+\w+\s+.*\bso\s+all\b|a\s+\w+\s+.*\bso\s+every\b|every\s+single\b|all\s+students\s+will|everyone\s+will\s+always|in\s+my\s+experience\s+.*(everyone|all))\b", text_lower))

        if label == "ad populum":
            return bool(re.search(r"\b(millions\s+of|everyone\s+uses|everybody\s+knows|most\s+people\s+agree|popular\s+opinion|majority\s+of\s+people|all\s+people\s+use)\b", text_lower))

        if label == "ad hominem":
            return bool(re.search(r"\b(you\s+(don't|do\s+not)\s+understand|your\s+opinion\s+.*doesn't\s+matter|because\s+you\s+are|you\s+have\s+no\s+(idea|clue|experience)|attack\s+on\s+the\s+person)\b", text_lower))

        if label == "circular reasoning":
            return bool(re.search(r"\b(because\s+it\s+is\s+.*therefore\s+it\s+is|true\s+because\s+it\s+is\s+true)\b", text_lower))

        if label == "appeal to emotion":
            return bool(re.search(r"\b(think\s+of\s+the\s+children|heartbreaking|terrifying|disastrous\s+tragedy|pity)\b", text_lower))

        # For other fallacies, require higher decision strength
        return decision_strength >= 0.70

    def predict_fallacy(
        self,
        text
    ):

        if self.logic_model is None:

            return {
                "label": None,
                "prediction": None,
                "decision_strength": 0.0,
                "reliable": False
            }

        try:

            raw_prediction = self.logic_model.predict(
                [text]
            )[0]

            decision_strength = 0.0

            if hasattr(
                self.logic_model,
                "decision_function"
            ):

                decision = (
                    self.logic_model.decision_function(
                        [text]
                    )
                )

                if len(decision.shape) == 1:

                    decision_strength = (
                        self._decision_strength(
                            decision[0]
                        )
                    )

                else:

                    maximum_margin = max(
                        abs(float(value))
                        for value in decision[0]
                    )

                    decision_strength = (
                        self._decision_strength(
                            maximum_margin
                        )
                    )

            label_str = str(raw_prediction).strip().lower()
            reliable = self.is_reliable_fallacy(
                fallacy_label=label_str,
                decision_strength=decision_strength,
                text=text
            )

            # If reliable, return the validated fallacy; otherwise None
            active_label = label_str if reliable else None

            return {
                "label": active_label,
                "prediction": active_label,
                "raw_prediction": label_str,
                "decision_strength": decision_strength,
                "reliable": reliable
            }

        except Exception as error:

            print(
                "Fallacy prediction error:",
                error
            )

            return {
                "label": None,
                "prediction": None,
                "decision_strength": 0.0,
                "reliable": False
            }

    # =========================================================
    # BIAS-AWARE LANGUAGE DETECTION
    # =========================================================

    def predict_bias(
        self,
        text
    ):

        if self.wnc_model is None:

            return {
                "label": "Not detected",
                "prediction": "0",
                "status": "not_detected",
                "decision_strength": 0.0
            }

        try:

            vectorizer = (
                self.wnc_model[
                    "vectorizer"
                ]
            )

            model = (
                self.wnc_model[
                    "model"
                ]
            )

            features = (
                vectorizer.transform(
                    [text]
                )
            )

            prediction = model.predict(
                features
            )[0]

            decision_strength = 0.0

            if hasattr(
                model,
                "decision_function"
            ):

                decision = (
                    model.decision_function(
                        features
                    )
                )

                decision_strength = (
                    self._decision_strength(
                        decision[0]
                    )
                )

            pred_str = str(prediction).strip()
            is_bias_class = pred_str in ["1", "biased", "bias-associated"]

            # Conservative bias tiers:
            # - not_detected: neutral or strength < 0.45
            # - possible: 0.45 <= strength < 0.65
            # - detected: strength >= 0.65
            if not is_bias_class or decision_strength < 0.45:
                status = "not_detected"
                display_label = "Not detected"
            elif decision_strength < 0.65:
                status = "possible"
                display_label = "Possible bias-associated phrasing"
            else:
                status = "detected"
                display_label = "Bias-associated language detected"

            return {
                "label": display_label,
                "prediction": "1" if status == "detected" else "0",
                "status": status,
                "decision_strength": decision_strength
            }

        except Exception as error:

            print(
                "Bias prediction error:",
                error
            )

            return {
                "label": "Not detected",
                "prediction": "0",
                "status": "not_detected",
                "decision_strength": 0.0
            }

    # =========================================================
    # SPECIFICITY SCORE
    # =========================================================

    def calculate_specificity(
        self,
        entities,
        statistics,
        dates
    ):

        score = 0.0

        if entities:
            score += 0.40

        if statistics:
            score += 0.35

        if dates:
            score += 0.25

        return round(
            min(score, 1.0),
            2
        )

    # =========================================================
    # SENTIMENT INTENSITY
    # =========================================================

    def calculate_sentiment_intensity(
        self,
        sentiment
    ):

        if not sentiment:
            return 0.0

        confidence = float(
            sentiment.get(
                "score",
                0.0
            )
        )

        label = str(
            sentiment.get(
                "label",
                "neutral"
            )
        ).lower()

        if label == "neutral":
            return round(
                (1.0 - confidence) * 0.5,
                2
            )

        return round(
            confidence,
            2
        )

    # =========================================================
    # CALCULATE FALLACY PENALTY
    # =========================================================

    def calculate_fallacy_penalty(
        self,
        fallacy_label,
        decision_strength,
        reliable: bool = True
    ):

        if not fallacy_label or not reliable:
            return 0.0

        label = str(fallacy_label).strip().lower()

        if label in ["", "none", "unknown", "intentional", "miscellaneous"]:
            return 0.0

        if decision_strength < 0.50:
            return 0.0

        if decision_strength < 0.70:
            return 7.0

        return 15.0

    # =========================================================
    # CALCULATE BIAS PENALTY
    # =========================================================

    def calculate_bias_penalty(
        self,
        bias_status,
        decision_strength
    ):

        # Only penalize when bias is conservatively DETECTED (>= 0.65)
        if bias_status != "detected":
            return 0.0

        if decision_strength < 0.75:
            return 5.0

        return 10.0

    # =========================================================
    # ARGUMENT QUALITY SCORE
    # =========================================================

    def calculate_quality_score(
        self,
        argument_component,
        evidence,
        fallacy,
        fallacy_strength,
        fallacy_reliable,
        bias_status,
        bias_strength,
        sentiment,
        specificity
    ):

        score = 50.0
        reasons = []

        # 1. Structure
        component = str(argument_component).strip().lower() if argument_component else ""

        if component in ["claim", "majorclaim", "premise"]:
            score += 15
            reasons.append("A recognizable argument component was detected.")
        else:
            reasons.append("Argument structure lacks formal premise/claim definition.")

        # 2. Evidence
        evidence_score = float(evidence.get("evidence_presence_score", 0.0))
        evidence_points = evidence_score * 20
        score += evidence_points

        if evidence_score >= 0.40 or evidence.get("is_evidence_bearing"):
            reasons.append("Supported by verifiable evidence signals or entities.")
        else:
            reasons.append("Evidence is limited; consider adding supporting examples or sources.")

        # 3. Fallacy Penalty (Only if reliable)
        fallacy_penalty = self.calculate_fallacy_penalty(
            fallacy_label=fallacy,
            decision_strength=fallacy_strength,
            reliable=fallacy_reliable
        )
        score -= fallacy_penalty

        if fallacy_penalty > 0 and fallacy:
            reasons.append(f"Potential fallacy detected: {fallacy} (-{int(fallacy_penalty)} pts).")

        # 4. Bias Penalty (Only if status == detected)
        bias_penalty = self.calculate_bias_penalty(
            bias_status=bias_status,
            decision_strength=bias_strength
        )
        score -= bias_penalty

        if bias_penalty > 0:
            reasons.append(f"Loaded bias-associated language detected (-{int(bias_penalty)} pts).")

        # 5. Specificity
        if specificity >= 0.40:
            score += 10
            reasons.append("Specific details, entities, or measurements provided.")

        score = max(0.0, min(100.0, score))

        return round(score, 1), reasons

    # =========================================================
    # QUALITY LEVEL
    # =========================================================

    def get_quality_level(
        self,
        score
    ):

        if score >= 75:
            return "Strong"
        elif score >= 50:
            return "Moderate"
        else:
            return "Needs Review"

    # =========================================================
    # COMPLETE ARGUMENT ANALYSIS
    # =========================================================

    def analyze(
        self,
        text
    ):

        if not text or not text.strip():
            return {
                "text": text,
                "argument_component": {},
                "fallacy": {},
                "bias": {},
                "sentiment": {},
                "evidence": {},
                "specificity_score": 0.0,
                "argument_quality_score": 0.0,
                "quality_level": "Needs Review",
                "explanation": []
            }

        argument_component = self.predict_argument_component(text)
        fallacy = self.predict_fallacy(text)
        bias = self.predict_bias(text)
        sentiment = self.sentiment_analyzer.analyze(text)
        evidence = self.evidence_analyzer.analyze(text)

        specificity = self.calculate_specificity(
            entities=evidence.get("entities", []),
            statistics=evidence.get("statistics", []),
            dates=evidence.get("dates", [])
        )

        score, reasons = self.calculate_quality_score(
            argument_component=argument_component.get("label"),
            evidence=evidence,
            fallacy=fallacy.get("prediction"),
            fallacy_strength=fallacy.get("decision_strength", 0.0),
            fallacy_reliable=fallacy.get("reliable", False),
            bias_status=bias.get("status", "not_detected"),
            bias_strength=bias.get("decision_strength", 0.0),
            sentiment=sentiment,
            specificity=specificity
        )

        return {
            "text": text,
            "argument_component": argument_component,
            "fallacy": fallacy,
            "bias": bias,
            "sentiment": sentiment,
            "evidence": evidence,
            "specificity_score": specificity,
            "argument_quality_score": score,
            "quality_level": self.get_quality_level(score),
            "explanation": reasons
        }