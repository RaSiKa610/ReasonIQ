from transformers import pipeline


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


print("Loading sentiment model...")
print(MODEL_NAME)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME
)


test_sentences = [
    "I completely agree with this argument.",
    "This policy is absolutely terrible.",
    "The evidence supports the proposed solution.",
    "I am not convinced by your argument.",
    "This is a reasonable approach.",
    "That claim is completely ridiculous.",
    "There are both advantages and disadvantages to this approach.",
]


print("\nSentiment Results")
print("=================")

for text in test_sentences:

    result = sentiment_pipeline(text)[0]

    print(f"\nText: {text}")
    print(f"Label: {result['label']}")
    print(f"Score: {result['score']:.4f}")