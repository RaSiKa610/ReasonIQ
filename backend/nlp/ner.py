import re
from transformers import pipeline


MODEL_NAME = "dslim/bert-base-NER"


class NERAnalyzer:
    def __init__(self):
        print("Loading NER model...")

        self.pipeline = pipeline(
            "token-classification",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            aggregation_strategy="simple"
        )

        print("NER model loaded successfully.")

        # Generic words / debate concepts that should not be presented as named entities
        self.generic_stopwords = {
            "ai", "a.i.", "artificial intelligence", "tech", "technology",
            "education", "high school", "high schools", "students", "student"
        }

    def _is_asr_artifact(self, entity_text: str, full_text: str, start: int, end: int) -> bool:
        """
        Check if entity is an artifact of broken tokenization or ASR corruption.
        E.g., 'Gan' extracted from 'Ganlones' or random non-alphabetic fragments.
        """
        # If the span in the original text is immediately adjacent to letters without boundary
        # e.g., 'Gan' followed immediately by 'lones' with no whitespace
        if end < len(full_text) and full_text[end].isalpha():
            return True
        if start > 0 and full_text[start - 1].isalpha():
            return True

        # Check for known ASR non-words
        if entity_text.lower() in ["gan", "ganlone", "ganlones", "thingy"]:
            return True

        return False

    def analyze(self, text):
        if not text or not text.strip():
            return []

        results = self.pipeline(text)

        entities = []

        for entity in results:
            start = entity["start"]
            end = entity["end"]
            ent_text = text[start:end].strip()

            if not ent_text:
                continue

            entities.append({
                "text": ent_text,
                "label": entity["entity_group"],
                "score": round(float(entity["score"]), 4),
                "start": start,
                "end": end
            })

        # Fix fragmented entities.
        # If two predictions touch each other and together form
        # a natural multi-word entity, keep the higher-confidence label.
        merged = []

        for entity in entities:

            if not merged:
                merged.append(entity)
                continue

            previous = merged[-1]

            # Check whether the two predictions touch/overlap or are separated by single space.
            touching = entity["start"] <= previous["end"] + 1

            if touching:

                # Select the label with higher confidence.
                if entity["score"] > previous["score"]:
                    selected_label = entity["label"]
                else:
                    selected_label = previous["label"]

                new_start = min(
                    previous["start"],
                    entity["start"]
                )

                new_end = max(
                    previous["end"],
                    entity["end"]
                )

                previous["text"] = text[new_start:new_end].strip()
                previous["label"] = selected_label
                previous["start"] = new_start
                previous["end"] = new_end

                previous["score"] = round(
                    max(
                        previous["score"],
                        entity["score"]
                    ),
                    4
                )

            else:
                merged.append(entity)

        # -----------------------------------------------------
        # Post-Processing & Filtering
        # -----------------------------------------------------
        filtered_entities = []
        seen_keys = set()

        for ent in merged:
            ent_text = ent["text"].strip()
            lower_ent = ent_text.lower()

            # 1. Skip generic concept 'AI'
            if lower_ent in self.generic_stopwords:
                continue

            # 2. Skip very short (< 2 chars) or empty entities
            if len(ent_text) < 2:
                continue

            # 3. Skip ASR artifacts or sub-word splits
            if self._is_asr_artifact(ent_text, text, ent["start"], ent["end"]):
                continue

            # 4. Deduplicate (prevent duplicate identical pills)
            key = (ent_text.lower(), ent["label"])
            if key in seen_keys:
                continue
            seen_keys.add(key)

            filtered_entities.append(ent)

        return filtered_entities


if __name__ == "__main__":

    analyzer = NERAnalyzer()

    test_sentences = [
        "According to WHO, India needs stronger healthcare policies.",
        "The government of Maharashtra announced a new renewable energy project.",
        "Elon Musk discussed artificial intelligence at an event in California.",
        "The United Nations released a report on climate change.",
        "In 2024, Maharashtra invested 500 crore rupees in renewable energy.",
        "Microsoft CEO Satya Nadella spoke at an event in Mumbai.",
        "students Ganlones faster with AI.",
        "OpenAI researchers presented the system in Mumbai.",
        "AI can help students."
    ]

    for text in test_sentences:

        print("\nText:", text)

        entities = analyzer.analyze(text)

        if not entities:
            print("No entities detected.")

        else:
            for entity in entities:
                print(
                    f"Entity: {entity['text']} | "
                    f"Label: {entity['label']} | "
                    f"Confidence: {entity['score']} | "
                    f"Span: {entity['start']}-{entity['end']}"
                )