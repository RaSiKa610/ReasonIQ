import re

from .ner import NERAnalyzer


class EvidenceExtractor:

    def __init__(self):
        print("Initializing Evidence Extractor...")

        self.ner = NERAnalyzer()

        print("Evidence Extractor ready.")

        # Phrases that commonly indicate that a speaker
        # is referring to evidence, research, data, or a source.
        self.evidence_indicators = [
            "according to",
            "research shows",
            "research suggests",
            "study found",
            "study shows",
            "studies show",
            "data shows",
            "data suggests",
            "evidence shows",
            "evidence suggests",
            "report states",
            "report says",
            "survey found",
            "survey shows",
            "experts say",
            "scientists say",
            "scientists found",
            "researchers found",
            "analysis shows",
            "statistics show"
        ]

        # Patterns used to identify explicitly mentioned sources.
        self.source_patterns = [
            r"according to\s+([^,.]+?)(?=\s+(?:,|and|found|reported|says|said|states|suggests|shows)|[,.]|$)",

            r"research from\s+([^,.]+?)(?=\s+(?:found|reported|shows|suggests|states)|[,.]|$)",

            r"study from\s+([^,.]+?)(?=\s+(?:found|reported|shows|suggests|states)|[,.]|$)",

            r"report by\s+([^,.]+?)(?=\s+(?:found|reported|shows|suggests|states)|[,.]|$)",

            r"reported by\s+([^,.]+?)(?=\s+(?:found|reported|shows|suggests|states)|[,.]|$)",

            r"published by\s+([^,.]+?)(?=\s+(?:found|reported|shows|suggests|states)|[,.]|$)"
        ]

    # ---------------------------------------------------------
    # STATISTICS
    # ---------------------------------------------------------

    def detect_statistics(self, text):

        patterns = [

            # Percentages
            r"\b\d+(?:\.\d+)?\s*%",

            # Numbers followed by percent
            r"\b\d+(?:\.\d+)?\s+percent\b",

            # Large numerical quantities
            r"\b\d+(?:\.\d+)?\s*(?:million|billion|trillion|crore|lakh)\b",

            # Currency values
            r"(?:₹|\$|€|£)\s*\d+(?:\.\d+)?\s*(?:million|billion|crore|lakh)?"
        ]

        statistics = []

        for pattern in patterns:

            matches = re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE
            )

            for match in matches:

                value = match.group().strip()

                if value not in statistics:
                    statistics.append(value)

        return statistics

    # ---------------------------------------------------------
    # DATES
    # ---------------------------------------------------------

    def detect_dates(self, text):

        dates = []

        # Four-digit years
        years = re.findall(
            r"\b(?:19|20)\d{2}\b",
            text
        )

        for year in years:

            if year not in dates:
                dates.append(year)

        # Date formats such as:
        # 15 September 2026
        # September 15, 2026

        date_patterns = [

            r"\b\d{1,2}\s+"
            r"(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)"
            r"\s+\d{4}\b",

            r"\b(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)"
            r"\s+\d{1,2},?\s+\d{4}\b"
        ]

        for pattern in date_patterns:

            matches = re.findall(
                pattern,
                text,
                flags=re.IGNORECASE
            )

            for date in matches:

                date = date.strip()

                if date not in dates:
                    dates.append(date)

        return dates

    # ---------------------------------------------------------
    # EVIDENCE INDICATORS
    # ---------------------------------------------------------

    def detect_evidence_indicators(self, text):

        text_lower = text.lower()

        found = []

        for indicator in self.evidence_indicators:

            if indicator in text_lower:
                found.append(indicator)

        return found

    # ---------------------------------------------------------
    # SOURCE EXTRACTION
    # ---------------------------------------------------------

    def detect_sources(self, text):

        sources = []

        for pattern in self.source_patterns:

            matches = re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE
            )

            for match in matches:

                source = match.group(1).strip()

                # Remove unnecessary leading/trailing spaces
                source = source.strip(" ,.")

                if source and source not in sources:
                    sources.append(source)

        return sources

    # ---------------------------------------------------------
    # EVIDENCE PRESENCE SCORE
    # ---------------------------------------------------------

    def calculate_evidence_presence_score(
        self,
        sources,
        statistics,
        dates,
        indicators,
        entities
    ):

        score = 0.0

        # Explicit evidence indicator
        if indicators:
            score += 0.30

        # Explicitly mentioned source
        if sources:
            score += 0.25

        # Numerical/statistical evidence
        if statistics:
            score += 0.20

        # Temporal information
        if dates:
            score += 0.10

        # Organizations or people may represent
        # referenced sources or authorities.
        source_entities = [
            entity
            for entity in entities
            if entity["label"] in ["ORG", "PER"]
        ]

        if source_entities:
            score += 0.15

        return round(min(score, 1.0), 2)

    # ---------------------------------------------------------
    # COMPLETE ANALYSIS
    # ---------------------------------------------------------

    def analyze(self, text):

        if not text or not text.strip():

            return {
                "text": text,
                "evidence_presence_score": 0.0,
                "is_evidence_bearing": False,
                "sources": [],
                "statistics": [],
                "dates": [],
                "indicators": [],
                "entities": []
            }

        # Named Entity Recognition
        entities = self.ner.analyze(text)

        # Numerical/statistical information
        statistics = self.detect_statistics(text)

        # Dates and years
        dates = self.detect_dates(text)

        # Evidence-related phrases
        indicators = self.detect_evidence_indicators(text)

        # Explicitly mentioned sources
        sources = self.detect_sources(text)

        # Calculate evidence presence
        evidence_presence_score = (
            self.calculate_evidence_presence_score(
                sources=sources,
                statistics=statistics,
                dates=dates,
                indicators=indicators,
                entities=entities
            )
        )

        return {
            "text": text,
            "evidence_presence_score": evidence_presence_score,
            "is_evidence_bearing": evidence_presence_score >= 0.40,
            "sources": sources,
            "statistics": statistics,
            "dates": dates,
            "indicators": indicators,
            "entities": entities
        }


# =============================================================
# TESTING
# =============================================================

if __name__ == "__main__":

    extractor = EvidenceExtractor()

    test_sentences = [

        "According to WHO, India reduced malaria cases by 40% in 2020.",

        "A 2024 study from Oxford University found that renewable energy reduced emissions.",

        "Research shows that 75 percent of students prefer online learning.",

        "The government announced a new education policy.",

        "I think this policy is absolutely terrible.",

        "According to the United Nations, global temperatures increased by 1.5 percent between 2015 and 2020.",

        "A report by Microsoft found that artificial intelligence adoption increased significantly."
    ]

    for text in test_sentences:

        print("\n" + "=" * 80)

        result = extractor.analyze(text)

        print("Text:")
        print(text)

        print("\nEvidence Presence Score:")
        print(result["evidence_presence_score"])

        print("\nEvidence Bearing:")
        print(result["is_evidence_bearing"])

        print("\nSources:")
        print(result["sources"])

        print("\nStatistics:")
        print(result["statistics"])

        print("\nDates:")
        print(result["dates"])

        print("\nEvidence Indicators:")
        print(result["indicators"])

        print("\nEntities:")

        for entity in result["entities"]:

            print(
                f"  {entity['text']} → "
                f"{entity['label']} "
                f"({entity['score']})"
            )