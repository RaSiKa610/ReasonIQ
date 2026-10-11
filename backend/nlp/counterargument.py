"""
ReasonIQ - Tactical Counterargument & Coaching Generator

Generates private coaching recommendations for debate participants based on
intervention status, reliable fallacy signals, evidence presence, and argumentation quality.
"""

from typing import Dict, Any, Optional


class CounterargumentGenerator:

    def __init__(self):

        print("Initializing Counterargument Generator...")

        self.fallacy_templates = {

            "ad populum":
                "Widespread agreement does not by itself establish whether a claim is correct. "
                "Consider examining the evidence and reasoning behind the position.",

            "ad hominem":
                "The statement appears to focus on a person rather than directly addressing "
                "the claim. Consider evaluating the argument using relevant evidence and reasoning.",

            "appeal to emotion":
                "Emotional language can influence how a claim is perceived without establishing "
                "whether it is correct. Consider examining the supporting evidence and reasoning.",

            "false causality":
                "The statement may assume a cause-and-effect relationship without establishing "
                "that relationship with evidence. Consider examining evidence that directly supports "
                "the claimed causal connection.",

            "false dilemma":
                "Consider whether there are additional options or intermediate positions "
                "between the extremes presented in this statement.",

            "faulty generalization":
                "The statement extends a conclusion from limited instances to a universal claim. "
                "Consider examining whether the evidence is broad and representative enough.",

            "circular reasoning":
                "The reasoning may use the conclusion itself as support. Consider providing "
                "independent evidence that supports the claim.",

            "equivocation":
                "The statement may use an important term in more than one sense. "
                "Clarifying the meaning of key terms could make the reasoning easier to evaluate.",

            "fallacy of relevance":
                "Some information in the statement may not directly support the main claim. "
                "Consider focusing on evidence that is directly relevant to the conclusion.",

            "fallacy of extension":
                "The statement may broaden or exaggerate another position. "
                "Consider addressing the original position directly before evaluating it.",

            "fallacy of credibility":
                "The statement may rely on the authority of a source rather than "
                "examining the claim itself. Consider evaluating the evidence independently.",

            "appeal to ignorance":
                "Lack of proof against a claim does not by itself prove that the claim is true. "
                "Consider examining whether positive, verifiable evidence supports the position.",

            "slippery slope":
                "The statement predicts an escalating sequence of extreme consequences. "
                "Consider examining whether each step in the chain is independently supported by evidence.",

            "false equivalence":
                "The statement equates two distinct concepts or activities. "
                "Consider whether material differences exist between them that affect the conclusion.",

            "straw man":
                "The statement may broaden or exaggerate another position. "
                "Consider addressing the original position directly before evaluating it.",

            "hasty generalization":
                "The statement extends a conclusion from limited instances to a universal claim. "
                "Consider examining whether the evidence is broad and representative enough.",

            "false cause":
                "The statement assumes a cause-and-effect relationship without establishing "
                "that relationship with evidence. Consider examining evidence that directly supports "
                "the claimed causal connection.",

            "bandwagon":
                "Widespread agreement does not by itself establish whether a claim is correct. "
                "Consider examining the evidence and reasoning behind the position.",

            "appeal to authority":
                "The statement may rely on the authority of a source rather than "
                "examining the claim itself. Consider evaluating the evidence independently."
        }

        self.bias_template = (
            "Some wording in the statement may be loaded or evaluative. "
            "Consider replacing subjective phrasing with neutral descriptions and supporting evidence."
        )

        self.evidence_template = (
            "What specific evidence, verifiable examples, or statistics support this claim?"
        )

        self.specificity_template = (
            "The statement is broad. Consider adding concrete details or examples to make the claim easier to evaluate."
        )

        self.general_template = (
            "Consider examining the claim using specific evidence, clear reasoning, "
            "and alternative explanations before reaching a conclusion."
        )

        print("Counterargument Generator ready.")

    # ---------------------------------------------------------
    # FALLACY
    # ---------------------------------------------------------

    def get_fallacy(self, fallacy_analysis):

        if not fallacy_analysis:
            return None, 0.0, False

        fallacy = str(
            fallacy_analysis.get("prediction") or fallacy_analysis.get("label") or ""
        ).strip().lower()

        try:
            strength = float(
                fallacy_analysis.get(
                    "decision_strength",
                    0.0
                )
            )
        except (TypeError, ValueError):
            strength = 0.0

        reliable = fallacy_analysis.get("reliable", True)

        return fallacy, strength, reliable

    def get_fallacy_response(self, fallacy_analysis):

        fallacy, strength, reliable = self.get_fallacy(
            fallacy_analysis
        )

        # Dataset labels that should not be treated as actionable fallacies.
        if not fallacy or fallacy in [
            "none",
            "unknown",
            "intentional",
            "miscellaneous"
        ]:
            return None

        # If flagged as unreliable, do NOT generate fallacy coaching
        if not reliable:
            return None

        # Only use a template when one exists.
        if fallacy not in self.fallacy_templates:
            return None

        return {
            "text": self.fallacy_templates[fallacy],
            "reason": (
                f"Potential {fallacy} signal detected "
                f"(decision strength: {strength:.2f})."
            ),
            "based_on": f"potential fallacy: {fallacy}",
            "strategy": "fallacy-focused reasoning"
        }

    # ---------------------------------------------------------
    # BIAS
    # ---------------------------------------------------------

    def get_bias_response(self, bias_analysis):

        if not bias_analysis:
            return None

        status = bias_analysis.get("status")
        if status != "detected":
            return None

        try:
            strength = float(
                bias_analysis.get(
                    "decision_strength",
                    0.0
                )
            )
        except (TypeError, ValueError):
            strength = 0.0

        return {
            "text": self.bias_template,
            "reason": (
                "Bias-associated wording detected "
                f"(decision strength: {strength:.2f})."
            ),
            "based_on": "bias-associated wording",
            "strategy": "bias-aware reframing"
        }

    # ---------------------------------------------------------
    # EVIDENCE
    # ---------------------------------------------------------

    def get_evidence_response(self, evidence_analysis):

        if not evidence_analysis:
            return None

        if evidence_analysis.get("status") == "not_applicable":
            return None

        try:
            score = float(
                evidence_analysis.get(
                    "evidence_presence_score",
                    0.0
                )
            )
        except (TypeError, ValueError):
            score = 0.0

        if score >= 0.40 or evidence_analysis.get("is_evidence_bearing"):
            return None

        return {
            "text": self.evidence_template,
            "reason": "Limited explicit supporting evidence detected.",
            "based_on": "limited evidence presence",
            "strategy": "evidence-focused reasoning"
        }

    # ---------------------------------------------------------
    # SPECIFICITY
    # ---------------------------------------------------------

    def get_specificity_response(self, quality_analysis):

        if not quality_analysis:
            return None

        try:
            score = float(
                quality_analysis.get(
                    "specificity_score",
                    0.0
                )
            )
        except (TypeError, ValueError):
            score = 0.0

        if score >= 0.40:
            return None

        return {
            "text": self.specificity_template,
            "reason": "The statement contains limited specific details.",
            "based_on": "low text specificity",
            "strategy": "specificity improvement"
        }

    # ---------------------------------------------------------
    # MAIN GENERATOR
    # ---------------------------------------------------------

    def generate(
        self,
        text,
        fallacy_analysis=None,
        bias_analysis=None,
        evidence_analysis=None,
        quality_analysis=None,
        intervention_analysis=None,
        is_social_statement=False
    ):

        if not text or not text.strip() or is_social_statement:
            return None

        # -----------------------------------------------------
        # PRIORITY 0: INTERVENTION / CONDUCT COACHING
        # -----------------------------------------------------

        if intervention_analysis and intervention_analysis.get("triggered"):
            itype = intervention_analysis.get("type")
            if itype == "personal_attack":
                return {
                    "counterargument": "Focus on the argument rather than the person. Explain which part of the opposing claim you disagree with and provide evidence or reasoning.",
                    "reason": "Personal attack detected. Address the argument rather than attacking the person.",
                    "strategy": "conduct-coaching",
                    "based_on": ["personal_attack"]
                }
            elif itype == "abusive_language":
                return {
                    "counterargument": "Please restate your disagreement respectfully and address the argument or evidence.",
                    "reason": "Abusive language detected. Maintain respectful debate conduct.",
                    "strategy": "conduct-coaching",
                    "based_on": ["abusive_language"]
                }
            elif itype == "insufficient_argument":
                return {
                    "counterargument": "Specify which problems you mean and explain why they matter, providing supporting reasoning.",
                    "reason": "The response does not provide sufficient argumentation.",
                    "strategy": "reasoning-expansion",
                    "based_on": ["insufficient_argument"]
                }
            elif itype == "unclear_statement":
                return {
                    "counterargument": "Try expressing the argument as a complete and specific statement.",
                    "reason": "The statement was unclear or fragmented.",
                    "strategy": "clarity-improvement",
                    "based_on": ["unclear_statement"]
                }
            elif itype == "low_information":
                return {
                    "counterargument": "Consider expanding your point with clear reasoning, evidence, or an explanation of your perspective.",
                    "reason": "The statement provided limited debate content.",
                    "strategy": "content-expansion",
                    "based_on": ["low_information"]
                }

        # -----------------------------------------------------
        # PRIORITY 1: RELIABLE FALLACY
        # -----------------------------------------------------

        fallacy_response = self.get_fallacy_response(
            fallacy_analysis
        )

        if fallacy_response:

            return {
                "counterargument":
                    fallacy_response["text"],

                "reason":
                    fallacy_response["reason"],

                "strategy":
                    fallacy_response["strategy"],

                "based_on":
                    [fallacy_response["based_on"]]
            }

        # -----------------------------------------------------
        # PRIORITY 2: EVIDENCE WEAKNESS
        # -----------------------------------------------------

        evidence_response = self.get_evidence_response(
            evidence_analysis
        )

        if evidence_response:

            return {
                "counterargument":
                    evidence_response["text"],

                "reason":
                    evidence_response["reason"],

                "strategy":
                    evidence_response["strategy"],

                "based_on":
                    [evidence_response["based_on"]]
            }

        # -----------------------------------------------------
        # PRIORITY 3: BIAS
        # -----------------------------------------------------

        bias_response = self.get_bias_response(
            bias_analysis
        )

        if bias_response:

            return {
                "counterargument":
                    bias_response["text"],

                "reason":
                    bias_response["reason"],

                "strategy":
                    bias_response["strategy"],

                "based_on":
                    [bias_response["based_on"]]
            }

        # -----------------------------------------------------
        # PRIORITY 4: SPECIFICITY
        # -----------------------------------------------------

        specificity_response = (
            self.get_specificity_response(
                quality_analysis
            )
        )

        if specificity_response:

            return {
                "counterargument":
                    specificity_response["text"],

                "reason":
                    specificity_response["reason"],

                "strategy":
                    specificity_response["strategy"],

                "based_on":
                    [specificity_response["based_on"]]
            }

        # -----------------------------------------------------
        # DEFAULT CONSTRUCTIVE COACHING
        # -----------------------------------------------------

        return {
            "counterargument":
                self.general_template,

            "reason":
                "General constructive coaching for debate reasoning.",

            "strategy":
                "general reasoning review",

            "based_on":
                ["general reasoning review"]
        }