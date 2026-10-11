from transformers import pipeline


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


class SentimentAnalyzer:

    def __init__(self):
        print("Loading sentiment model...")

        self.pipeline = pipeline(
            "sentiment-analysis",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME
        )

        print("Sentiment model loaded successfully.")

    def analyze(self, text):

        if not text or not text.strip():
            return {
                "label": "neutral",
                "score": 0.0
            }

        result = self.pipeline(text)[0]

        return {
            "label": result["label"].lower(),
            "score": round(result["score"], 4)
        }


if __name__ == "__main__":

    analyzer = SentimentAnalyzer()

    test_text = [
        "I completely agree with this argument.",
        "This policy is absolutely terrible.",
        "The evidence supports the proposed solution.",
        "That claim is completely ridiculous."
    ]

    for text in test_text:

        result = analyzer.analyze(text)

        print("\nText:", text)
        print("Sentiment:", result["label"])
        print("Confidence:", result["score"])