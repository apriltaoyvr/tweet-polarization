import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.clean import clean_text # noqa: E402

LABELS = ["agreement", "disagreement"]
RAW = ["nli_agrees", "nli_disagrees",
       "sent_negative", "sent_positive", "irony"]

MODEL_NAME = "all-MiniLM-L6-v2"

df = pd.read_csv(
    ROOT / "data/interim/features.csv",
    dtype={"reply_id": str, "response_to": str},
)

if "op_claim" not in df.columns:
    tweets = pd.read_csv(ROOT / "data/raw/tweets.csv",
                         dtype={"tweet_id": str})
    claim_map = dict(zip(tweets["tweet_id"], tweets["claim"]))
    df["op_claim"] = df["response_to"].map(claim_map)

df = df[df["personal_label"].isin(LABELS)].copy()
df[RAW] = df[RAW].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=RAW + ["reply_text", "response_to", "op_claim"]).copy()

counts = df["personal_label"].value_counts()
print(counts, "\n")
if len(counts) < 2 or counts.min() < 3:
    raise SystemExit("Need at least 2 classes with 3+ examples each.")
print(pd.crosstab(df["response_to"], df["personal_label"]), "\n")

df["reply_text"] = df["reply_text"].map(clean_text)

encoder = SentenceTransformer(MODEL_NAME)
text_emb = encoder.encode(
    df["reply_text"].tolist(), normalize_embeddings=True)

emb_cols = [f"emb_{i}" for i in range(text_emb.shape[1])]
X = pd.concat(
    [
        pd.DataFrame(text_emb, columns=emb_cols, index=df.index),
        df[RAW].astype(float),
    ],
    axis=1,
)
y = df["personal_label"]
y_bin = (y == "disagreement").astype(int)
groups = df["response_to"]

preprocess = ColumnTransformer([
    ("emb", StandardScaler(), emb_cols),
    ("scores", StandardScaler(), RAW),
])

model = Pipeline(
    [
        ("prep", preprocess),
        (
            "clf",
            LogisticRegression(
                C=0.01,
                solver="liblinear",
                max_iter=2000,
                class_weight="balanced",
            ),
        ),
    ]
)

cv = LeaveOneGroupOut()


# distinguish agreement vs. disagreement within the same thread
def within_tweet_auc(y_bin, p, groups):
    y_bin, p, groups = map(np.asarray, (y_bin, p, groups))
    num = den = 0.0
    for t in np.unique(groups):
        m = groups == t
        pos = y_bin[m].sum()
        neg = m.sum() - pos
        if pos and neg:
            num += roc_auc_score(y_bin[m], p[m]) * pos * neg
            den += pos * neg
    return num / den


for name, est in [
    ("majority baseline", DummyClassifier(strategy="most_frequent")),
    ("joint emb + raw scores (C=0.01)", model),
]:
    pred = cross_val_predict(est, X, y, groups=groups, cv=cv)
    pred_bin = (pred == "disagreement").astype(int)
    print(
        f"== {name} ==  macro-F1 = {f1_score(y, pred, average='macro', zero_division=0):.3f}"
    )
    print(
        f"balanced accuracy = {balanced_accuracy_score(y_bin, pred_bin):.3f}")
    if hasattr(est, "predict_proba") and name != "majority baseline":
        proba = cross_val_predict(
            est, X, y, groups=groups, cv=cv, method="predict_proba"
        )
        print(f"ROC AUC           = {roc_auc_score(y_bin, proba[:, 1]):.3f}")
        print(
            f"within-tweet AUC  = {within_tweet_auc(y_bin, proba[:, 1], groups):.3f}")
    print(classification_report(y, pred, zero_division=0))
    print(confusion_matrix(y, pred, labels=LABELS), "\n")

model.fit(X, y)

(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(
    {
        "pipeline": model,
        "embedding_model": MODEL_NAME,
        "emb_cols": emb_cols,
        "raw_cols": RAW,
    },
    ROOT / "models/stance_classifier.joblib",
)
print("Saved models/stance_classifier.joblib")
