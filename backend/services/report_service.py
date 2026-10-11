"""
ReasonIQ - Final Debate Performance Report Service

Generates authoritative, explainable end-of-debate performance reports (0–100)
for each speaker upon debate conclusion.
Includes traceable deduction audit trails, dimension breakdowns, speaking history,
personalized strengths, what to avoid, and coaching recommendations.

Correctly isolates social statements (e.g. "Thank you") so they do not
distort substantive argument quality averages.
"""

from typing import Dict, Any, List, Optional
from services.room_manager import RoomState, UtteranceState, ParticipantState


class ReportService:
    """
    Calculates final scores and generates structured performance reports.
    """

    @staticmethod
    def generate_speaker_report(
        room: RoomState,
        participant: ParticipantState
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive report for a specific participant in a debate.
        Aggregates authoritative completed utterances.
        """
        p_id = participant.participant_id
        
        # Authoritative completed utterances belonging to this participant
        completed_utterances = [
            u for u in room.utterances
            if (
                u.participant_id == p_id or
                (getattr(u, "speaker_name", None) == participant.name and getattr(u, "speaker_label", None) == participant.speaker_label)
            )
            and u.status == "COMPLETED"
            and u.nlp_analysis is not None
        ]

        total_turns = len(completed_utterances)

        # ----------------------------------------------------
        # Metrics Aggregation
        # ----------------------------------------------------
        interventions_count = 0
        personal_attacks_count = 0
        abusive_count = 0
        unclear_count = 0
        insufficient_count = 0
        evidence_present_count = 0
        fallacies_count = 0

        deductions: List[Dict[str, Any]] = []
        base_scores: List[float] = []
        final_turn_scores: List[float] = []
        evidence_scores: List[float] = []
        clarity_scores: List[float] = []
        conduct_scores: List[float] = []

        turn_history: List[Dict[str, Any]] = []

        if total_turns == 0:
            return {
                "participant_id": participant.participant_id,
                "speaker_label": participant.speaker_label or "Participant",
                "speaker_name": participant.name,
                "room_code": room.room_code,
                "topic": room.topic,
                "total_turns": 0,
                "analyzed_turns": 0,
                "base_score": None,
                "intervention_penalty": 0,
                "intervention_penalties": 0,
                "final_score": None,
                "performance_tier": "No Analyzed Turns",
                "performance_level": "No Analyzed Turns",
                "summary": {
                    "total_turns": 0,
                    "analyzed_turns": 0,
                    "interventions_count": 0,
                    "personal_attacks": 0,
                    "unclear_statements": 0,
                    "insufficient_arguments": 0,
                    "evidence_signals": 0
                },
                "score_breakdown": None,
                "breakdown": None,
                "deductions": [],
                "strengths": [],
                "areas_to_improve": [],
                "how_to_improve": [],
                "recommendations": [],
                "what_to_avoid": [],
                "message": "No completed speaking turns were available for analysis.",
                "turn_history": [],
                "speaking_history": []
            }

        for index, utt in enumerate(completed_utterances, start=1):
            nlp = utt.nlp_analysis or {}
            quality = nlp.get("quality", {})
            intervention = nlp.get("intervention", {})
            evidence = nlp.get("evidence", {})
            fallacy = nlp.get("fallacy", {})
            counterarg = nlp.get("counterargument", {})
            content_type = nlp.get("content_type", "substantive_argument")

            # Check if this is a non-substantive social statement
            is_social = (content_type == "social_statement") or (quality.get("score") is None)

            if not is_social:
                turn_base_score = float(quality.get("base_score", 50.0) if quality.get("base_score") is not None else 50.0)
                turn_penalty = float(intervention.get("penalty", 0.0) if intervention.get("triggered") else 0.0)
                turn_final_score = float(quality.get("score", max(0.0, turn_base_score - turn_penalty)))

                base_scores.append(turn_base_score)
                final_turn_scores.append(turn_final_score)

                # Evidence Tracking
                ev_score = float(evidence.get("evidence_presence_score", 0.0))
                if evidence.get("is_evidence_bearing") or ev_score >= 0.35:
                    evidence_present_count += 1
                evidence_scores.append(min(100.0, ev_score * 100.0 + (30.0 if evidence.get("entities") else 0.0)))

                # Fallacy Tracking
                if fallacy and fallacy.get("prediction") and str(fallacy.get("prediction")).lower() not in ["none", "valid", ""]:
                    fallacies_count += 1

                # Clarity & Conduct sub-scores for this turn
                itype = intervention.get("type") if intervention.get("triggered") else None
                if itype == "unclear_statement":
                    turn_clarity = 45.0
                elif itype == "insufficient_argument":
                    turn_clarity = 60.0
                else:
                    turn_clarity = float(quality.get("clarity_score", 90.0) or 90.0)

                turn_conduct = 100.0 - (turn_penalty * 3.0)
                clarity_scores.append(max(0.0, min(100.0, turn_clarity)))
                conduct_scores.append(max(0.0, min(100.0, turn_conduct)))

                t_base_disp = round(turn_base_score, 1)
                t_pen_disp = round(turn_penalty, 1)
                t_fin_disp = round(turn_final_score, 1)
            else:
                turn_penalty = 0.0
                t_base_disp = None
                t_pen_disp = 0.0
                t_fin_disp = None

            # Audit Deductions
            if intervention.get("triggered"):
                interventions_count += 1
                itype = intervention.get("type", "conduct_issue")
                if itype == "personal_attack":
                    personal_attacks_count += 1
                elif itype == "abusive_language":
                    abusive_count += 1
                elif itype == "unclear_statement":
                    unclear_count += 1
                elif itype == "insufficient_argument":
                    insufficient_count += 1

                deductions.append({
                    "turn_number": index,
                    "utterance_id": utt.utterance_id,
                    "type": itype,
                    "category": itype,
                    "points": int(turn_penalty),
                    "text": utt.transcript or "(No transcript)",
                    "statement": utt.transcript or "(No transcript)",
                    "warning": intervention.get("warning", ""),
                    "reason": intervention.get("details", intervention.get("warning", "Conduct penalty"))
                })

            fallacy_pen = float(quality.get("fallacy_penalty", 0.0) or 0.0)
            if fallacy_pen > 0:
                deductions.append({
                    "turn_number": index,
                    "utterance_id": utt.utterance_id,
                    "type": "fallacy_penalty",
                    "category": fallacy.get("prediction", "logical_fallacy"),
                    "points": int(fallacy_pen),
                    "text": utt.transcript or "(No transcript)",
                    "statement": utt.transcript or "(No transcript)",
                    "warning": fallacy.get("label", "Logical Fallacy Signal"),
                    "reason": fallacy.get("explanation", f"Point deduction applied for {fallacy.get('label', 'logical fallacy')}.")
                })

            bias_pen = float(quality.get("bias_penalty", 0.0) or 0.0)
            if bias_pen > 0:
                deductions.append({
                    "turn_number": index,
                    "utterance_id": utt.utterance_id,
                    "type": "bias_penalty",
                    "category": "bias_associated_language",
                    "points": int(bias_pen),
                    "text": utt.transcript or "(No transcript)",
                    "statement": utt.transcript or "(No transcript)",
                    "warning": "Bias-associated language",
                    "reason": "Point deduction applied for bias-associated phrasing."
                })

            # Turn history entry
            turn_history.append({
                "turn_number": index,
                "utterance_id": utt.utterance_id,
                "participant_id": utt.participant_id,
                "speaker_name": utt.speaker_name,
                "speaker_label": utt.speaker_label,
                "created_at": utt.created_at,
                "timestamp": utt.created_at,
                "transcript": utt.transcript,
                "status": utt.status,
                "content_type": content_type,
                "base_score": t_base_disp,
                "penalty": t_pen_disp,
                "final_score": t_fin_disp,
                "quality_score": t_fin_disp,
                "intervention": intervention if intervention.get("triggered") else None,
                "public_analysis": {
                    "content_type": content_type,
                    "argument_component": nlp.get("argument_component"),
                    "fallacy": nlp.get("fallacy"),
                    "bias": nlp.get("bias"),
                    "sentiment": nlp.get("sentiment"),
                    "evidence": nlp.get("evidence"),
                    "ner": nlp.get("ner") or (nlp.get("evidence", {}) or {}).get("entities", []),
                    "quality": quality
                },
                # Private counterargument coaching only
                "private_counterargument": counterarg
            })

        # ----------------------------------------------------
        # Final Score & Dimensional Calculations (0-100)
        # ----------------------------------------------------
        analyzed_substantive_turns = len(final_turn_scores)

        if analyzed_substantive_turns > 0:
            avg_final_score = sum(final_turn_scores) / len(final_turn_scores)
            avg_base_score = sum(base_scores) / len(base_scores)
            avg_evidence_score = sum(evidence_scores) / len(evidence_scores) if evidence_scores else 50.0
            avg_clarity_score = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 80.0
            avg_conduct_score = sum(conduct_scores) / len(conduct_scores) if conduct_scores else 100.0

            final_score_rounded = round(max(0.0, min(100.0, avg_final_score)), 1)
            base_score_rounded = round(max(0.0, min(100.0, avg_base_score)), 1)
            evidence_score_rounded = round(max(0.0, min(100.0, avg_evidence_score)), 1)
            clarity_score_rounded = round(max(0.0, min(100.0, avg_clarity_score)), 1)
            conduct_score_rounded = round(max(0.0, min(100.0, avg_conduct_score)), 1)

            if final_score_rounded >= 75.0:
                performance_tier = "Excellent"
            elif final_score_rounded >= 60.0:
                performance_tier = "Proficient"
            elif final_score_rounded >= 45.0:
                performance_tier = "Moderate"
            else:
                performance_tier = "Needs Development"

            breakdown_dict = {
                "argument_quality": base_score_rounded,
                "evidence": evidence_score_rounded,
                "evidence_quality": evidence_score_rounded,
                "clarity": clarity_score_rounded,
                "clarity_and_structure": clarity_score_rounded,
                "debate_conduct": conduct_score_rounded
            }
        else:
            final_score_rounded = None
            base_score_rounded = None
            performance_tier = "No Substantive Turns"
            breakdown_dict = None

        # ----------------------------------------------------
        # Qualitative Explainable Feedback
        # ----------------------------------------------------
        strengths: List[str] = []
        what_to_avoid: List[str] = []
        areas_to_improve: List[str] = []

        if personal_attacks_count == 0 and abusive_count == 0:
            strengths.append("Maintained respectful, professional debate conduct throughout all speaking turns.")
        if evidence_present_count >= max(1, total_turns // 2):
            strengths.append("Consistently supported claims with recognizable evidence and named entities.")
        if clarity_scores and (sum(clarity_scores) / len(clarity_scores)) >= 80.0:
            strengths.append("Expressed arguments clearly with well-formed sentence structure.")
        if fallacies_count == 0 and total_turns > 0:
            strengths.append("Avoided common logical reasoning traps and presented sound arguments.")
        if not strengths:
            strengths.append("Actively participated in the debate and presented distinct debate positions.")

        if personal_attacks_count > 0:
            what_to_avoid.append("Avoid attacking opponents personally; focus strictly on evaluating their claims and evidence.")
            areas_to_improve.append("Reframe disagreements by addressing the flaw in the opposing argument rather than criticizing the speaker.")
        if abusive_count > 0:
            what_to_avoid.append("Avoid using abusive, profane, or derogatory language in formal debate.")
        if unclear_count > 0:
            what_to_avoid.append("Avoid fragmented or incomplete sentences that obscure your intended argument.")
            areas_to_improve.append("Formulate complete, structured claims before speaking to improve clarity.")
        if insufficient_count > 0:
            what_to_avoid.append("Avoid one-word assertions or unsupported dismissals (e.g. 'Obviously', 'No').")
            areas_to_improve.append("Always accompany agreements or disagreements with an explanatory 'because...' premise.")
        if evidence_present_count < max(1, total_turns // 2) and analyzed_substantive_turns > 0:
            areas_to_improve.append("Incorporate specific verifiable examples, statistics, or documented sources to strengthen claims.")

        if not what_to_avoid:
            what_to_avoid.append("Avoid relying on general assertions without introducing specific supporting examples.")
        if not areas_to_improve:
            areas_to_improve.append("Continue deepening argument depth by exploring counter-perspectives proactively.")

        total_deductions_points = round(sum(d.get("points", 0) for d in deductions), 1)

        return {
            "participant_id": participant.participant_id,
            "speaker_label": participant.speaker_label or "Participant",
            "speaker_name": participant.name,
            "room_code": room.room_code,
            "topic": room.topic,
            "total_turns": total_turns,
            "analyzed_turns": total_turns,
            "analyzed_substantive_turns": analyzed_substantive_turns,
            "base_score": base_score_rounded,
            "intervention_penalty": total_deductions_points,
            "intervention_penalties": total_deductions_points,
            "final_score": final_score_rounded,
            "performance_tier": performance_tier,
            "performance_level": performance_tier,
            "summary": {
                "total_turns": total_turns,
                "analyzed_turns": total_turns,
                "interventions_count": interventions_count,
                "personal_attacks": personal_attacks_count,
                "unclear_statements": unclear_count,
                "insufficient_arguments": insufficient_count,
                "evidence_signals": evidence_present_count
            },
            "score_breakdown": breakdown_dict,
            "breakdown": breakdown_dict,
            "deductions": deductions,
            "strengths": strengths,
            "areas_to_improve": areas_to_improve,
            "how_to_improve": areas_to_improve,
            "recommendations": areas_to_improve,
            "what_to_avoid": what_to_avoid,
            "turn_history": turn_history,
            "speaking_history": turn_history
        }

    @staticmethod
    def sanitize_report_for_public(report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strips private coaching counterarguments from turn/speaking history
        so other participants cannot access private opponent coaching.
        """
        public_report = dict(report)
        sanitized_history = []
        for turn in report.get("turn_history", []):
            t = dict(turn)
            t.pop("private_counterargument", None)
            sanitized_history.append(t)
        public_report["turn_history"] = sanitized_history
        public_report["speaking_history"] = sanitized_history
        return public_report


# Singleton instance
report_service = ReportService()
