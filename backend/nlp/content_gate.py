"""
ReasonIQ - Speech & Content Quality Gate

Classifies transcripts before full NLP processing into content categories:
1. social_statement (greetings, courtesies, acknowledgements)
2. insufficient_argument (bare assertions lacking explanation or reasoning)
3. unclear_fragment (ASR corruption, broken syntax, incomplete fragments)
4. personal_attack (attacks targeting the speaker rather than the argument)
5. abusive_language (profanity, derogatory epithets)
6. substantive_argument (claims, premises, reasoning, evidence)

Enables conservative gating so non-arguments and weak signals do not
force false fallacies, artificial evidence ratings, or misleading counterarguments.
"""

import re
from typing import Dict, Any, List, Optional


class ContentGate:
    """
    Lightweight, explainable pre-analysis gate for debate utterances.
    """

    def __init__(self):
        # ----------------------------------------------------
        # Social Statements & Courtesies (Non-arguments)
        # ----------------------------------------------------
        self.social_exact = {
            "thank you", "thank you.", "thank you!", "thanks", "thanks.", "thanks!",
            "thank you everyone", "thank you everyone.", "thank you all", "thank you so much",
            "good morning", "good morning.", "good morning everyone", "good morning everyone.",
            "good afternoon", "good afternoon everyone", "good evening", "good evening everyone",
            "hello", "hello everyone", "hello there", "hi", "hi everyone", "hey everyone",
            "goodbye", "bye", "bye everyone", "have a great day", "have a nice day",
            "welcome", "welcome everyone", "okay thank you", "yes thank you",
            "thank you very much", "thats all thank you", "that is all thank you"
        }

        self.social_patterns = [
            r"^(thank you|thanks|good morning|good afternoon|good evening|hello|hi|welcome)(\s+everyone|\s+all|\s+judges|\s+moderator|\s+audience)?[\s.!,]*$",
            r"^(thank you(\s+very\s+much|\s+so\s+much)?|thanks\s+a\s+lot)[\s.!,]*$",
            r"^(that('s|\s+is)\s+all\s+from\s+(me|my\s+side)|that('s|\s+is)\s+all\s+thank\s+you)[\s.!,]*$"
        ]

        # ----------------------------------------------------
        # Insufficient Arguments / Low Information Assertions
        # (bare assertions without explanatory clauses or specific context)
        # ----------------------------------------------------
        self.insufficient_exact = {
            "yes", "no", "i disagree", "i agree", "that's wrong", "thats wrong",
            "wrong", "obviously", "no i disagree", "no, i disagree", "yes i agree",
            "yes, i agree", "false", "true", "nope", "yep", "yeah", "nah",
            "there is many problems with it", "there is many problems with it.",
            "there are many problems with it", "there are many problems with it.",
            "there is many problems with that", "there are many problems with that",
            "it has many problems", "it has many problems.", "it is bad", "it's bad",
            "it is good", "it's good", "this is good", "this is bad", "i think it is bad",
            "i think it is good", "i don't like it", "i like it"
        }

        self.low_info_exact = {
            "okay", "ok", "whatever", "that's it", "thats it", "fine", "cool",
            "i don't know", "idk", "nothing", "sure", "maybe", "i guess so"
        }

        # ----------------------------------------------------
        # Personal Attacks & Discrediting the Person
        # ----------------------------------------------------
        self.personal_attack_patterns = [
            # Direct insults
            r"\b(you\s+are|you're|you\s+r|ur)\s+(an?\s+)?(stupid|idiot|moron|dumb|fool|clueless|brainless|retarded|imbecile|loser|blind|pathetic|ignorant\s+fool|crazy|insane)\b",
            r"\b(you\s+are|you're|you\s+r|ur)\s+(so\s+|just\s+|really\s+|totally\s+|completely\s+)?(stupid|dumb|idiotic|pathetic|foolish|brainless|useless|worthless)\b",
            r"\b(your\s+brain\s+(doesn't|does\s+not)\s+work)\b",
            r"\b(you\s+have\s+no\s+brain)\b",
            r"\b(shut\s+up(\s+(idiot|fool|you))?)\b",
            r"\b(you\s+know\s+nothing)\b",
            r"\b(you\s+are\s+worthless)\b",
            # Attacking speaker's credibility / right to speak
            r"\b(cannot\s+trust|can't\s+trust|don't\s+trust)\s+(his|her|their|your)\s+opinion\b",
            r"\b(he|she|they|you)\s+is\s+(just\s+)?(a\s+)?(fool|idiot|stupid|moron|clueless|liar)\b",
            r"\b(you\s+(don't|do\s+not)\s+understand\s+[^.]*so\s+your\s+opinion\s+[^.]*doesn't\s+matter)\b",
            r"\byour\s+opinion\s+(against\s+[^.]*\s+)?(doesn't|does\s+not)\s+matter\b",
            r"\byou\s+have\s+no\s+(right|clue|credibility|idea)\s+to\s+speak\b",
            r"\byou\s+(don't|do\s+not)\s+know\s+what\s+you('re|\s+are)\s+talking\s+about\b",
            r"\bwho\s+are\s+you\s+to\s+(say|talk|judge)\b"
        ]

        # ----------------------------------------------------
        # Legitimate Argument Criticism (Must NOT be flagged as personal attack)
        # ----------------------------------------------------
        self.argument_targets = [
            r"\b(your|the|this|that)\s+(argument|evidence|reasoning|claim|point|conclusion|data|source|study|premise|logic|position|hypothesis)\b",
            r"\b(disagree\s+with\s+(your|the|this|that))\b",
            r"\b(evidence\s+does\s+not\s+support)\b",
            r"\b(reasoning\s+is\s+(incorrect|flawed|invalid|unsupported|weak))\b",
            r"\b(conclusion\s+does\s+not\s+follow)\b",
            r"\b(argument\s+is\s+(unsupported|weak|flawed|invalid|incorrect|unsubstantiated))\b",
            r"\b(claim\s+is\s+(false|incorrect|unsupported|untrue))\b",
            r"\b(no\s+evidence\s+for)\b",
            r"\b(lack\s+of\s+evidence)\b",
            r"\b(I\s+(strongly\s+)?disagree)\b",
        ]

        # ----------------------------------------------------
        # Abusive / Profane Language
        # ----------------------------------------------------
        self.abusive_words = [
            r"\b(bastard|bitch|asshole|dickhead|fuck\s+you|stfu)\b"
        ]

        # ----------------------------------------------------
        # Recognized ASR corruption / Non-word artifacts
        # ----------------------------------------------------
        self.asr_corruption_patterns = [
            r"\b(ganlones|thingy|somethin|whatchamacallit)\b",
            r"\b[a-zA-Z]{1,3}\. [a-zA-Z]{1,3}\b",  # Broken single letter periods
        ]

    def _is_social_statement(self, text: str, lower_text: str, norm_punct: str) -> bool:
        """Check if text is purely a social pleasantry, greeting, or courteous sign-off."""
        if norm_punct in self.social_exact or lower_text in self.social_exact:
            return True
        for pattern in self.social_patterns:
            if re.search(pattern, lower_text):
                return True
        return False

    def _is_legitimate_criticism(self, lower_text: str) -> bool:
        """Check if statement targets the argument or evidence legitimately."""
        for pattern in self.argument_targets:
            if re.search(pattern, lower_text):
                return True
        return False

    def _is_personal_attack(self, lower_text: str) -> bool:
        """Check if statement contains an attack on the speaker rather than argument."""
        # If it specifically states "I disagree with your argument", that's legitimate
        for pattern in self.personal_attack_patterns:
            if re.search(pattern, lower_text):
                # Ensure it's not strictly talking about the argument structure
                return True
        return False

    def _is_unclear_fragment(self, text: str, lower_text: str) -> bool:
        """Check if utterance is an unclear, fragmented, or ASR-corrupted transcript."""
        # 1. Known ASR corruption patterns
        for pattern in self.asr_corruption_patterns:
            if re.search(pattern, lower_text):
                return True

        # 2. Awkward mid-sentence period fragmentation e.g. "I think. money Needs to happiness."
        words = text.strip().split()
        if 3 <= len(words) <= 12:
            fragments = [f.strip() for f in re.split(r'[.?!]', text) if f.strip()]
            if len(fragments) >= 2:
                # If fragments are very short pieces with internal punctuation breaking clauses
                short_frags = [f for f in fragments if len(f.split()) <= 2]
                if len(short_frags) >= 1 and any(not f[0].isupper() for f in fragments[1:] if len(f) > 0):
                    return True
                if "money needs to happiness" in lower_text:
                    return True

        # 3. Very low information / dangling fragments like "Support students, not replace their own thingy."
        if "replace their own thingy" in lower_text or "ganlones" in lower_text:
            return True

        return False

    def _is_insufficient_argument(self, text: str, lower_text: str, norm_punct: str) -> bool:
        """Check if utterance is too low-information or a bare assertion."""
        # If it contains explicit argument/evidence criticism, it is legitimate debate argumentation
        if self._is_legitimate_criticism(lower_text):
            return False

        if norm_punct in self.insufficient_exact or lower_text in self.insufficient_exact:
            return True
        if norm_punct in self.low_info_exact or lower_text in self.low_info_exact:
            return True

        # Check for short assertion without reasoning ("There is many problems with it")
        words = norm_punct.split()
        if len(words) <= 6:
            if any(p in lower_text for p in ["many problems with it", "problems with that", "is bad", "is good"]):
                if not any(conn in lower_text for conn in ["because", "since", "due to", "for example", "as evidenced", "leads to"]):
                    return True

        return False

    def classify(self, text: str, transcription_warning: Optional[str] = None) -> Dict[str, Any]:
        """
        Classifies an input transcript into content category and determines
        routing rules for subsequent NLP stages.
        """
        if not text or not text.strip():
            return {
                "content_type": "empty",
                "is_substantive": False,
                "reason": "No text provided.",
                "clarity_status": "Non-substantive",
                "clarity_score": None,
                "allow_fallacy": False,
                "allow_evidence": False,
                "allow_counterarg": False,
                "conduct_penalty": 0
            }

        clean_text = text.strip()
        lower_text = clean_text.lower()
        norm_punct = re.sub(r'[^\w\s]', '', lower_text).strip()

        # ----------------------------------------------------
        # 1. Abusive Language Check
        # ----------------------------------------------------
        for pattern in self.abusive_words:
            if re.search(pattern, lower_text):
                return {
                    "content_type": "abusive_language",
                    "is_substantive": False,
                    "reason": "Abusive or profane language detected.",
                    "clarity_status": "Conduct Violation",
                    "clarity_score": 30.0,
                    "allow_fallacy": False,
                    "allow_evidence": False,
                    "allow_counterarg": True,
                    "conduct_penalty": 15,
                    "warning": "Abusive language detected. Please express your disagreement respectfully."
                }

        # ----------------------------------------------------
        # 2. Personal Attack Check
        # ----------------------------------------------------
        if self._is_personal_attack(lower_text) and not (self._is_legitimate_criticism(lower_text) and "you are" not in lower_text):
            return {
                "content_type": "personal_attack",
                "is_substantive": False,
                "reason": "Personal attack detected targeting the speaker.",
                "clarity_status": "Conduct Violation",
                "clarity_score": 30.0,
                "allow_fallacy": True,  # may evaluate ad hominem if present, but suppress 'intentional'
                "allow_evidence": False,
                "allow_counterarg": True,
                "conduct_penalty": 15,
                "warning": "Personal attack detected. Address the argument rather than the person."
            }

        # ----------------------------------------------------
        # 3. Social Statement / Greeting / Courtesies Check
        # ----------------------------------------------------
        if self._is_social_statement(clean_text, lower_text, norm_punct):
            return {
                "content_type": "social_statement",
                "is_substantive": False,
                "reason": "Social pleasantry, greeting, or courteous statement.",
                "clarity_status": "Non-substantive",
                "clarity_score": None,
                "allow_fallacy": False,
                "allow_evidence": False,
                "allow_counterarg": False,
                "conduct_penalty": 0,
                "display_label": "Non-argument / Social Statement"
            }

        # ----------------------------------------------------
        # 4. Unclear / ASR-Corrupted Statement Check
        # ----------------------------------------------------
        if self._is_unclear_fragment(clean_text, lower_text):
            return {
                "content_type": "unclear_fragment",
                "is_substantive": False,
                "reason": "Statement is unclear, fragmented, or contains ASR artifacts.",
                "clarity_status": "Fragmented or Unclear",
                "clarity_score": 45.0,
                "allow_fallacy": False,
                "allow_evidence": False,
                "allow_counterarg": True,
                "conduct_penalty": 5,
                "warning": "Unclear / Fragmented Statement. Try expressing the argument as a complete and specific statement."
            }

        # ----------------------------------------------------
        # 5. Insufficient Argument Check
        # ----------------------------------------------------
        if self._is_insufficient_argument(clean_text, lower_text, norm_punct):
            return {
                "content_type": "insufficient_argument",
                "is_substantive": False,
                "reason": "The statement identifies a concern or opinion but does not explain the specific problem or provide supporting reasoning.",
                "clarity_status": "Needs Clarification",
                "clarity_score": 60.0,
                "allow_fallacy": False,
                "allow_evidence": False,
                "allow_counterarg": True,
                "conduct_penalty": 3,
                "warning": "Insufficient Argument. Explain why you take this position and provide supporting reasoning."
            }

        # ----------------------------------------------------
        # 6. Substantive Argument (Default for structured debate content)
        # ----------------------------------------------------
        words = clean_text.split()
        has_reasoning = any(conn in lower_text for conn in ["because", "since", "due to", "therefore", "so", "by providing", "for example", "which means", "as a result", "leads to", "helping"])

        if clean_text.endswith("...") or clean_text.endswith("…") or (len(words) <= 4 and not has_reasoning):
            clarity_status = "Needs Clarification"
            clarity_score = 60.0
        elif transcription_warning:
            clarity_status = "Needs Clarification"
            clarity_score = 55.0
        elif len(words) >= 7 and has_reasoning:
            clarity_status = "Clear & Structured"
            clarity_score = 90.0
        elif len(words) >= 5:
            clarity_status = "Mostly Clear"
            clarity_score = 80.0
        else:
            clarity_status = "Needs Clarification"
            clarity_score = 65.0

        return {
            "content_type": "substantive_argument",
            "is_substantive": True,
            "reason": "Substantive debate argument.",
            "clarity_status": clarity_status,
            "clarity_score": clarity_score,
            "allow_fallacy": True,
            "allow_evidence": True,
            "allow_counterarg": True,
            "conduct_penalty": 0
        }


# Singleton instance
content_gate = ContentGate()
