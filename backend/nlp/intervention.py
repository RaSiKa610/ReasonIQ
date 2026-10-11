"""
ReasonIQ - Speech Quality & Debate Intervention Layer

Identifies debate conduct issues (personal attacks, abusive language)
and speech quality concerns (unclear/fragmented statements, insufficient arguments)
with explainable penalties and coaching prompts.

Distinguishes attacks on the *person* from legitimate criticism of
the *argument*, *evidence*, or *reasoning*.
"""

import re
from typing import Dict, Any, Optional
from .content_gate import content_gate


# ============================================================
# Centralized Configurable Intervention Penalties
# ============================================================

PERSONAL_ATTACK_PENALTY = 15
ABUSIVE_LANGUAGE_PENALTY = 15
UNCLEAR_STATEMENT_PENALTY = 5
INSUFFICIENT_ARGUMENT_PENALTY = 3
LOW_INFORMATION_PENALTY = 3


class InterventionDetector:
    """
    Analyzes debate utterances for conduct violations and speech quality issues.
    """

    def __init__(self):
        self.gate = content_gate

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Run intervention analysis on the input text via the content quality gate.
        """
        if not text or not text.strip():
            return {
                "triggered": False,
                "type": None,
                "severity": "none",
                "warning": "",
                "penalty": 0,
                "details": "No text provided."
            }

        gate_res = self.gate.classify(text)
        ctype = gate_res.get("content_type")

        if ctype == "abusive_language":
            return {
                "triggered": True,
                "type": "abusive_language",
                "severity": "high",
                "warning": "Abusive language detected. Please express your disagreement respectfully.",
                "penalty": ABUSIVE_LANGUAGE_PENALTY,
                "details": "The statement contains abusive or profane language."
            }

        elif ctype == "personal_attack":
            return {
                "triggered": True,
                "type": "personal_attack",
                "severity": "high",
                "warning": "Personal attack detected. Address the argument rather than the person.",
                "penalty": PERSONAL_ATTACK_PENALTY,
                "details": "The statement targets the individual instead of addressing their argument or evidence."
            }

        elif ctype == "unclear_fragment":
            return {
                "triggered": True,
                "type": "unclear_statement",
                "severity": "medium",
                "warning": "Unclear / Fragmented Statement. Try expressing the argument as a complete and specific statement.",
                "penalty": UNCLEAR_STATEMENT_PENALTY,
                "details": "The statement has broken structure or fragmented syntax."
            }

        elif ctype == "insufficient_argument":
            return {
                "triggered": True,
                "type": "insufficient_argument",
                "severity": "low",
                "warning": "Insufficient argument detected. Explain why you take this position and provide supporting reasoning.",
                "penalty": INSUFFICIENT_ARGUMENT_PENALTY,
                "details": "Statement identifies an assertion without supporting reasoning."
            }

        return {
            "triggered": False,
            "type": None,
            "severity": "none",
            "warning": "",
            "penalty": 0,
            "details": "Normal debate contribution."
        }


# Singleton instance
intervention_detector = InterventionDetector()
