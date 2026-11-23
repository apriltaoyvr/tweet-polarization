import os

import pandas as pd
import joblib
from transformers import pipeline
from features import RAW

tweets = pd.read_csv("data/raw/tweets.csv",
                     dtype={"tweet_id": str}, index_col=False)
replies = pd.read_csv("data/raw/replies.csv",
                      dtype={"tweet_id": str, "original_tweet": str}, index_col=False)
merged = replies.merge(
    tweets[["tweet_id", "tweet_text", "claim"]],
    left_on="original_tweet", right_on="tweet_id", suffixes=("", "_op"),
)

sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    top_k=None,
)
nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None
               )
irony_model = pipeline("text-classification",
                       model="cardiffnlp/twitter-roberta-base-irony",
                       top_k=None
                       )


def nli_scores(premise: str, hypothesis: str) -> dict:
    scores = nli({"text": premise, "text_pair": hypothesis}, truncation=True)
    return {s["label"].lower(): s["score"] for s in scores}


def stance(reply_text, claim, margin=0.1):
    scores = nli_scores(reply_text, claim)
    agree = scores["entailment"]
    disagree = scores["contradiction"]

    if abs(agree - disagree) < margin:
        label = "neutral"
    else:
        label = "agreement" if agree > disagree else "disagreement"
    return label, agree, disagree


def sentiment_scores(text: str) -> dict:
    scores = sentiment_model(text)[0]
    return {s["label"].lower(): s["score"] for s in scores}


def irony_score(text: str) -> float:
    scores = irony_model(text)[0]
    return next(s["score"] for s in scores if s["label"] == "irony")


def generate_features():
    output = []

    for _, r in merged.iterrows():
        text, claim = r["reply_text"], r["claim"]

        sent = sentiment_scores(text)
        irony = round(irony_score(text), 3)
        _, agree, disagree = stance(text, claim)

        output.append({
            "reply_id": r["tweet_id"],
            "response_to": r["original_tweet"],
            "reply_text": text,
            "nli_agrees": round(agree, 3),
            "nli_disagrees": round(disagree, 3),
            "sent_negative": round(sent["negative"], 3),
            "sent_neutral": round(sent["neutral"], 3),
            "sent_positive": round(sent["positive"], 3),
            "irony": irony,
            "personal_label": r.get("personal_label", "") if not pd.isna(r.get("personal_label")) else "",
        })

    return output


if __name__ == "__main__":
    df = pd.DataFrame(generate_features())

    clf = joblib.load("models/stance_classifier.joblib")
    df["predicted_stance"] = clf.predict(df[RAW])

    os.makedirs("data/interim", exist_ok=True)
    df.to_csv("data/interim/features.csv", index=False)
    print(f"Saved {len(df)} rows to data/interim/features.csv")
