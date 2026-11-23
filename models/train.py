import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LABELS = ["agreement", "disagreement", "neutral"]

feats = pd.read_csv("../data/interim/features.csv",
                    dtype={"reply_id": str, "response_to": str})
tweets = pd.read_csv("../data/raw/tweets.csv", dtype={"tweet_id": str})[
    ["tweet_id", "political_leaning"]]
df = feats.merge(tweets, left_on="response_to",
                 right_on="tweet_id", how="left")
df = df[df["personal_label"].isin(LABELS)].copy()

counts = df["personal_label"].value_counts()
print(counts, "\n")
if len(counts) < 2 or counts.min() < 3:
    raise SystemExit(
        "Need at least 2 classes with 3+ examples each. Label more rows.")


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


df["nli_agrees_logit"] = logit(df["nli_agrees"])
df["nli_disagrees_logit"] = logit(df["nli_disagrees"])
df["neg_x_irony"] = df["sent_negative"] * df["irony"]

numeric = ["nli_agrees_logit", "nli_disagrees_logit", "sent_negative",
           "sent_positive", "irony", "neg_x_irony"]
categorical = ["political_leaning"]

X = df[numeric + categorical]
y = df["personal_label"]
groups = df["response_to"]

pre = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
])
model = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced")),
])

cv = LeaveOneGroupOut()

print(pd.crosstab(df["response_to"], df["personal_label"]), "\n")

for name, est in [("majority baseline", DummyClassifier(strategy="most_frequent")),
                  ("logistic regression", model)]:
    pred = cross_val_predict(est, X, y, groups=groups, cv=cv)
    print(f"== {name} ==  macro-F1 = {f1_score(y, pred, average='macro'):.3f}")
    print(classification_report(y, pred, zero_division=0))
    print(confusion_matrix(y, pred, labels=LABELS), "\n")

model.fit(X, y)

names = model.named_steps["pre"].get_feature_names_out()
clf = model.named_steps["clf"]
rows = list(clf.classes_) if len(clf.classes_) > 2 else [
    f"{clf.classes_[1]} (vs {clf.classes_[0]})"]
coefs = pd.DataFrame(clf.coef_, index=rows, columns=names)
print(coefs.round(2).T)
joblib.dump(model, "stance_classifier.joblib")
