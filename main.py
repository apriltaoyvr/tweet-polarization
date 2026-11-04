import os

from transformers import pipeline

sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    top_k=1,
)

nli_model = pipeline(
    "text-classification",
    model="facebook/bart-large-mnli",
    top_k=1
)

zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def get_sentiment(text: str) -> str:
    scores = sentiment_model(text)[0]
    return max(scores, key=lambda s: s["score"])["label"]

def get_stance(tweet: str, reply: str) -> dict:
    text = f'Tweet: "{tweet}"\nReply: "{reply}"'
    labels = [
        "the reply agrees with the tweet",
        "the reply disagrees with the tweet",
        "the reply is neutral or unrelated to the tweet",
    ]
    result = zero_shot(text, candidate_labels=labels)
    top = result["labels"][0]
    stance = {"agrees": "agrees", "disagrees": "disagrees"}.get(
        top.split("reply ")[1].split(" ")[0], "neither"
    )
    return {"stance": stance, "confidence": round(result["scores"][0], 3)}

tweet = "This new tax policy will hurt small businesses."
 
replies = [
    "Exactly, this is a disaster for my shop.",
    "Great, finally some sanity in government!",
    "lol you clearly don't understand economics",
    "Does anyone have a source for this?",
]

if __name__ == "__main__":
    print(f"TWEET: {tweet}\n")
    for reply in replies:
        sentiment = get_sentiment(reply)
        stance = get_stance(tweet, reply)
        print(f"REPLY:     {reply}")
        print(f"  sentiment: {sentiment}")
        print(f"  stance:    {stance['stance']} (confidence {stance['confidence']})")
        print()
