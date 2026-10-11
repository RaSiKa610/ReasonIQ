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
    # CONVERT SVM DECISION TO CALIBRATED STRENGTH
    # =========================================================

    def _decision_strength(
        self,
        decision
    ):
        """
        Converts an SVM decision margin into a calibrated 0-1 strength value.
        Positive margins indicate confidence in the predicted class.
        Negative margins indicate the sample is far from the decision boundary
        for this class and yield calibrated strength < 0.50.
        """
        try:
            import math
            value = float(decision)
            # Sigmoid calibration: 1 / (1 + exp(-1.5 * margin))
            prob = 1.0 / (1.0 + math.exp(-1.5 * value))
            return round(prob, 4)
        except Exception:
            return 0.0

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

            decision_strength = 0.50

            if hasattr(self.aaec_model, "decision_function"):
                decision = self.aaec_model.decision_function([text])
                if len(decision.shape) == 1:
                    decision_strength = self._decision_strength(decision[0])
                else:
                    if hasattr(self.aaec_model, "classes_"):
                        classes = list(self.aaec_model.classes_)
                        if prediction in classes:
                            pred_idx = classes.index(prediction)
                            margin = float(decision[0][pred_idx])
                            decision_strength = self._decision_strength(margin)
                        else:
                            decision_strength = self._decision_strength(max(decision[0]))
                    else:
                        decision_strength = self._decision_strength(max(decision[0]))

            return {
                "label": str(prediction),
                "decision_strength": decision_strength
            }

        except Exception as error:
            print("Argument component prediction error:", error)
            return {
                "label": "unknown",
                "decision_strength": 0.0
            }

    # =========================================================
    # FALLACY DETECTION (Hybrid: Model + Explainable Reasoning Patterns)
    # =========================================================

    def predict_fallacy(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Hybrid fallacy detector covering all 12 key fallacy categories:
        1. Hasty generalization
        2. Circular reasoning
        3. Straw man
        4. False equivalence
        5. Appeal to ignorance
        6. False cause
        7. Slippery slope
        8. Ad hominem
        9. Bandwagon / appeal to popularity
        10. False dilemma
        11. Appeal to authority
        12. Appeal to emotion / other categories

        Uses calibrated SVM margins combined with explainable reasoning patterns.
        Conservative decision thresholds prevent forcing fallacies on ordinary
        disagreements, negative sentiment, or general opinions.
        """
        if not text or not text.strip():
            return {
                "label": "No strong signal detected",
                "prediction": None,
                "raw_prediction": "none",
                "decision_strength": 0.0,
                "reliable": False,
                "explanation": ""
            }

        text_lower = text.lower().strip()

        # Check for legitimate debate criticism first - never label as ad hominem or fallacy
        is_legitimate_criticism = bool(re.search(
            r"\b((your|the|this|that)\s+(argument|evidence|reasoning|claim|point|data|logic)\s+does\s+not\s+support|"
            r"evidence\s+is\s+(insufficient|weak|lacking|flawed)|disagree\s+with\s+(that|the|your)\s+(proposal|claim|argument)|"
            r"sample\s+size\s+is\s+too\s+small)\b",
            text_lower
        ))

        # Check for bare insult without reasoning (handled by conduct intervention, not reasoning fallacy)
        is_bare_insult = bool(re.search(r"^(you\s+are|you're)\s+(stupid|an?\s+idiot|a\s+fool|dumb|worthless)[\s.!,]*$", text_lower))

        # Get raw SVM model prediction and calibrated margin
        svm_pred = "none"
        svm_strength = 0.0
        if self.logic_model is not None:
            try:
                raw = self.logic_model.predict([text])[0]
                svm_pred = str(raw).strip().lower()
                if hasattr(self.logic_model, "decision_function"):
                    dec = self.logic_model.decision_function([text])
                    if hasattr(self.logic_model, "classes_"):
                        classes = list(self.logic_model.classes_)
                        if raw in classes:
                            pred_idx = classes.index(raw)
                            margin = float(dec[0][pred_idx])
                        else:
                            margin = float(max(dec[0]))
                    else:
                        margin = float(dec[0])
                    svm_strength = self._decision_strength(margin)
            except Exception as e:
                print("SVM prediction error:", e)

        # ----------------------------------------------------
        # EXPLAINABLE REASONING PATTERN MATCHING (Hybrid Layer)
        # ----------------------------------------------------

        detected_fallacy = None
        detected_label = None
        detected_strength = 0.0
        detected_explanation = ""

        # 1. BANDWAGON / APPEAL TO POPULARITY
        if not is_legitimate_criticism and (
            re.search(r"\b(everyone\s+uses\s+.*\bso\s+.*(better|superior|must\s+be)|everybody\s+knows\s+.*\bso\b|millions\s+of\s+people\s+(use|agree)\s+.*\bso\b|majority\s+of\s+people\s+agree\s+.*\btherefore\b|popular\s+opinion\s+proves)\b", text_lower)
            or (svm_pred == "ad populum" and any(w in text_lower for w in ["everyone", "everybody", "millions of", "popular", "majority"]))
        ):
            detected_fallacy = "ad populum"
            detected_label = "Ad populum / Bandwagon"
            detected_strength = max(0.80, svm_strength)
            detected_explanation = "Popularity or widespread adoption does not by itself establish superiority or truth."

        # 2. FALSE DILEMMA
        elif not is_legitimate_criticism and (
            re.search(r"\b(either\s+we\s+.*or\s+we\s+(completely\s+)?(destroy|lose|fail|ruin|fall\s+behind)|only\s+two\s+(options|choices)|must\s+choose\s+between\s+.*or)\b", text_lower)
            or (svm_pred == "false dilemma" and re.search(r"\b(either\b.*\bor\b|if\s+(we\s+)?(don't|do\s+not)\b.*(never|no\s+future))\b", text_lower))
        ):
            detected_fallacy = "false dilemma"
            detected_label = "False dilemma"
            detected_strength = max(0.85, svm_strength)
            detected_explanation = "The statement presents two extreme alternatives as though no intermediate options exist."

        # 3. AD HOMINEM (Attack on person's traits/credibility)
        elif not is_legitimate_criticism and not is_bare_insult and (
            re.search(r"\b(cannot\s+trust\s+(his|her|their|your)\s+opinion\s+.*because\s+(he|she|they|you)\s+(is|are)\s+(just\s+)?(a\s+)?(fool|idiot|stupid|moron|clueless|biased|liar)|your\s+opinion\s+.*doesn't\s+matter\s+because\s+you\s+are|attack\s+on\s+the\s+person)\b", text_lower)
            or (svm_pred == "ad hominem" and svm_strength >= 0.65 and "opinion" in text_lower)
        ):
            detected_fallacy = "ad hominem"
            detected_label = "Ad hominem"
            detected_strength = max(0.85, svm_strength)
            detected_explanation = "Attacking the speaker's personal character or credibility rather than evaluating their argument."

        # 4. APPEAL TO IGNORANCE
        elif not is_legitimate_criticism and re.search(
            r"\b((nobody|no\s+one|no-one)\s+has\s+(proved|proven|shown|demonstrated)\s+that\s+.*\bso\b|(there\s+is\s+no|hasn't\s+been\s+any)\s+proof\s+that\s+.*\bso\b|(has\s+not|hasn't)\s+been\s+(disproved|disproven)\s+so\b|cannot\s+prove\s+.*false\s+so\b)\b",
            text_lower
        ):
            detected_fallacy = "appeal to ignorance"
            detected_label = "Appeal to ignorance"
            detected_strength = 0.82
            detected_explanation = "Treating the absence of disproof as evidence of truth or superiority."

        # 5. SLIPPERY SLOPE (Checked before hasty generalization for multi-step cascading claims)
        elif not is_legitimate_criticism and re.search(
            r"\b(if\s+(one|we\s+allow|we\s+let|\w+)\s+.*soon\s+.*(nobody|everyone\s+will|all\s+will)\s+.*and\s+.*(useless|destroyed|collapse|ruined|disappear\s+forever|no\s+future))\b",
            text_lower
        ):
            detected_fallacy = "slippery slope"
            detected_label = "Slippery slope"
            detected_strength = 0.80
            detected_explanation = "Asserting an inevitable chain of extreme consequences from an initial action without supporting evidence."

        # 6. HASTY GENERALIZATION
        elif not is_legitimate_criticism and (
            re.search(r"\b((one|a\s+single|a\s+few)\s+\w+\s+.*\bso\s+(all|every|everyone|everybody)\b|in\s+my\s+experience\s+.*(everyone|all)|saw\s+one\s+.*so\s+all)\b", text_lower)
            or (svm_pred == "faulty generalization" and any(p in text_lower for p in ["one ", "a single ", "in my experience", "all students", "all people"]))
        ):
            detected_fallacy = "faulty generalization"
            detected_label = "Hasty generalization"
            detected_strength = max(0.80, svm_strength)
            detected_explanation = "Drawing a broad or universal conclusion from an isolated or unrepresentative example."

        # 7. FALSE CAUSE
        elif not is_legitimate_criticism and (
            re.search(r"\b(started\s+.*\band\s+.*(dropped|fell|rose|increased|decreased)\s*,\s*so\s+.*(must\s+have\s+caused|caused|is\s+the\s+reason)|happened\s+.*\bso\s+.*must\s+have\s+caused\b)\b", text_lower)
            or (svm_pred == "false causality" and svm_strength >= 0.55)
        ):
            detected_fallacy = "false causality"
            detected_label = "False cause"
            detected_strength = max(0.78, svm_strength)
            detected_explanation = "Inferring a causal link from sequential occurrence or association without sufficient supporting evidence."

        # 8. CIRCULAR REASONING
        elif not is_legitimate_criticism and (
            re.search(r"\b(better\s+because\s+.*(superior|better)|superior\s+because\s+.*(better|superior)|good\s+because\s+.*(good|great)|true\s+because\s+.*(true|fact)|wrong\s+because\s+.*(wrong|incorrect)|because\s+it\s+is\s+.*therefore\s+it\s+is|true\s+because\s+it\s+is\s+true)\b", text_lower)
            or (svm_pred == "circular reasoning" and svm_strength >= 0.50)
        ):
            detected_fallacy = "circular reasoning"
            detected_label = "Circular reasoning"
            detected_strength = max(0.78, svm_strength)
            detected_explanation = "The conclusion is restated as its own supporting premise without offering independent evidence."

        # 9. STRAW MAN
        elif not is_legitimate_criticism and (
            re.search(r"\b(you\s+(just\s+)?want\s+to\s+(ban\s+all|destroy\s+all|eliminate\s+all|outlaw\s+all|stop\s+all|get\s+rid\s+of\s+all)|they\s+want\s+to\s+(ban\s+all|destroy\s+all)|so\s+you('re|\s+are)\s+saying\s+(we\s+should\s+ban|nobody\s+should|all\s+.*must\s+be\s+banned))\b", text_lower)
            or (svm_pred == "fallacy of extension" and svm_strength >= 0.55)
        ):
            detected_fallacy = "fallacy of extension"
            detected_label = "Straw man"
            detected_strength = max(0.75, svm_strength)
            detected_explanation = "Misrepresenting or exaggerating an opposing position to make it easier to attack."

        # 10. FALSE EQUIVALENCE
        elif not is_legitimate_criticism and (
            re.search(r"\b(is\s+(exactly|precisely|just|literally)\s+the\s+same\s+as|is\s+no\s+different\s+(than|from)|is\s+equivalent\s+to|amounts\s+to\s+the\s+(exact\s+)?same\s+thing\s+as)\b", text_lower)
            or (svm_pred in ["equivocation", "fallacy of relevance"] and any(p in text_lower for p in ["same as", "no different than", "equivalent to"]))
        ):
            detected_fallacy = "equivocation"
            detected_label = "False equivalence"
            detected_strength = max(0.76, svm_strength)
            detected_explanation = "Treating materially different activities or concepts as identical without supporting justification."

        # 11. APPEAL TO AUTHORITY
        elif not is_legitimate_criticism and (
            re.search(r"\b(experts\s+say\s+.*so\s+it\s+cannot\s+be\s+questioned|celebrity\s+says\s+.*therefore\s+it\s+is\s+fact|authority\s+says\s+so\s+so\s+it\s+must\s+be\s+true)\b", text_lower)
            or (svm_pred == "fallacy of credibility" and svm_strength >= 0.65)
        ):
            detected_fallacy = "fallacy of credibility"
            detected_label = "Appeal to authority"
            detected_strength = max(0.75, svm_strength)
            detected_explanation = "Treating authority or status as definitive proof rather than providing underlying reasoning and evidence."

        # 12. APPEAL TO EMOTION
        elif not is_legitimate_criticism and (
            re.search(r"\b(think\s+of\s+the\s+children|heartbreaking\s+tragedy\s+proves|terrifying\s+danger\s+demands)\b", text_lower)
            or (svm_pred == "appeal to emotion" and svm_strength >= 0.65)
        ):
            detected_fallacy = "appeal to emotion"
            detected_label = "Appeal to emotion"
            detected_strength = max(0.72, svm_strength)
            detected_explanation = "Using emotionally charged rhetoric in place of factual evidence or reasoning."

        # Fallback to model if confidence is genuinely high and not an artifact
        elif (
            svm_pred not in ["", "none", "unknown", "intentional", "miscellaneous"]
            and svm_strength >= 0.70
            and not is_legitimate_criticism
            and not is_bare_insult
        ):
            detected_fallacy = svm_pred
            detected_label = svm_pred.capitalize()
            detected_strength = svm_strength
            detected_explanation = f"Potential {svm_pred} reasoning pattern detected."

        # Output formatting based on calibrated confidence
        if detected_fallacy and detected_strength >= 0.45:
            prefix = "Possible " if detected_strength < 0.70 else ""
            final_label = f"{prefix}{detected_label} signal" if not detected_label.endswith("signal") else detected_label
            return {
                "prediction": detected_fallacy,
                "label": final_label,
                "display_name": detected_label,
                "raw_prediction": svm_pred,
                "decision_strength": detected_strength,
                "reliable": True,
                "explanation": detected_explanation
            }

        return {
            "prediction": None,
            "label": "No strong signal detected",
            "display_name": "No strong signal detected",
            "raw_prediction": svm_pred,
            "decision_strength": 0.0,
            "reliable": False,
            "explanation": ""
        }

    # =========================================================
    # BIAS-AWARE LANGUAGE DETECTION
    # =========================================================

    def predict_bias(
        self,
        text
    ):
        if not text or not text.strip():
            return {
                "label": "Not detected",
                "prediction": "0",
                "status": "not_detected",
                "decision_strength": 0.0
            }

        text_lower = text.lower().strip()
        # Legitimate debate criticism or simple disagreement is not bias
        if re.search(r"\b((your|the|this|that)\s+(argument|evidence|reasoning|claim|point|data|logic)\s+does\s+not\s+support|evidence\s+is\s+(insufficient|weak|lacking|flawed)|disagree\s+with\s+(that|the|your)\s+(proposal|claim|argument)|strongly\s+disagree)\b", text_lower):
            return {
                "label": "Not detected",
                "prediction": "0",
                "status": "not_detected",
                "decision_strength": 0.0
            }

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
        specificity,
        text: str = ""
    ):
        base_score = 50.0
        reasons = []

        # 1. Structure
        component = str(argument_component).strip().lower() if argument_component else ""
        if component in ["claim", "majorclaim", "premise"]:
            base_score += 12.0
            reasons.append("Recognizable argument structure detected.")
        else:
            reasons.append("Argument structure lacks formal premise/claim definition.")

        # Reasoning connectives
        text_lower = text.lower() if text else ""
        has_reasoning = any(conn in text_lower for conn in ["because", "since", "due to", "therefore", "so ", "for example", "as a result", "leads to"])
        if has_reasoning:
            base_score += 6.0
            reasons.append("Clear explanatory connective present (e.g. because, since, therefore).")

        # 2. Evidence
        evidence_score = float(evidence.get("evidence_presence_score", 0.0))
        evidence_points = round(evidence_score * 18.0, 1)
        base_score += evidence_points

        if evidence_score >= 0.35 or evidence.get("is_evidence_bearing"):
            reasons.append("Supported by verifiable evidence signals or entities.")
        else:
            reasons.append("Evidence is limited; consider adding supporting examples or sources.")

        # 3. Specificity
        if specificity >= 0.40:
            spec_pts = round(specificity * 12.0, 1)
            base_score += spec_pts
            reasons.append("Specific details, entities, or measurements provided.")

        base_score = max(35.0, min(100.0, base_score))

        # 4. Fallacy Penalty (Only if reliable, and ad hominem handled without duplicate)
        fallacy_penalty = 0.0
        if fallacy and fallacy_reliable and fallacy_strength >= 0.50:
            fallacy_str = str(fallacy).lower()
            # Do NOT apply fallacy penalty for ad hominem here because personal attack
            # is penalized centrally in intervention detector (-15 pts) to prevent double counting
            if fallacy_str != "ad hominem":
                fallacy_penalty = 10.0 if fallacy_strength >= 0.70 else 6.0
                reasons.append(f"Potential fallacy signal detected: {fallacy} (-{int(fallacy_penalty)} pts).")

        # 5. Bias Penalty (Only if status == detected)
        bias_penalty = self.calculate_bias_penalty(
            bias_status=bias_status,
            decision_strength=bias_strength
        )
        if bias_penalty > 0:
            reasons.append(f"Loaded bias-associated language detected (-{int(bias_penalty)} pts).")

        # Final quality score after reasoning deductions (intervention conduct penalties applied in engine)
        score = max(0.0, min(100.0, base_score - fallacy_penalty - bias_penalty))

        return round(base_score, 1), round(score, 1), round(fallacy_penalty, 1), round(bias_penalty, 1), reasons

    # =========================================================
    # QUALITY LEVEL
    # =========================================================

    def get_quality_level(
        self,
        score
    ):
        if score is None:
            return "Unavailable"
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
                "base_quality_score": 0.0,
                "argument_quality_score": 0.0,
                "fallacy_penalty": 0.0,
                "bias_penalty": 0.0,
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

        base_score, score, fallacy_penalty, bias_penalty, reasons = self.calculate_quality_score(
            argument_component=argument_component.get("label"),
            evidence=evidence,
            fallacy=fallacy.get("prediction"),
            fallacy_strength=fallacy.get("decision_strength", 0.0),
            fallacy_reliable=fallacy.get("reliable", False),
            bias_status=bias.get("status", "not_detected"),
            bias_strength=bias.get("decision_strength", 0.0),
            sentiment=sentiment,
            specificity=specificity,
            text=text
        )

        return {
            "text": text,
            "argument_component": argument_component,
            "fallacy": fallacy,
            "bias": bias,
            "sentiment": sentiment,
            "evidence": evidence,
            "specificity_score": specificity,
            "base_quality_score": base_score,
            "argument_quality_score": score,
            "fallacy_penalty": fallacy_penalty,
            "bias_penalty": bias_penalty,
            "quality_level": self.get_quality_level(score),
            "explanation": reasons
        }