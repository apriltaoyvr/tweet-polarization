import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parent
if not (ROOT / "features.py").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from features import RAW, NUMERIC, add_engineered   # noqa: E402

LABELS = ["agreement", "disagreement", "neutral"]

df = pd.read_csv(ROOT / "data/interim/features.csv",
                 dtype={"reply_id": str, "response_to": str})
df = df[df["personal_label"].isin(LABELS)].copy()

counts = df["personal_label"].value_counts()
print(counts, "\n")
if len(counts) < 2 or counts.min() < 3:
    raise SystemExit("Need at least 2 classes with 3+ examples each.")
print(pd.crosstab(df["response_to"], df["personal_label"]), "\n")

X = df[RAW]                     # raw columns only; engineering happens inside the model
y = df["personal_label"]
groups = df["response_to"]

model = Pipeline([
    ("fe", FunctionTransformer(add_engineered)),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced")),
])

cv = LeaveOneGroupOut()
for name, est in [("majority baseline", DummyClassifier(strategy="most_frequent")),
                  ("logistic regression", model)]:
    pred = cross_val_predict(est, X, y, groups=groups, cv=cv)
    print(f"== {name} ==  macro-F1 = {f1_score(y, pred, average='macro'):.3f}")
    print(classification_report(y, pred, zero_division=0))
    print(confusion_matrix(y, pred, labels=LABELS), "\n")

model.fit(X, y)

clf = model.named_steps["clf"]
rows = (list(clf.classes_) if len(clf.classes_) > 2
        else [f"{clf.classes_[1]} (vs {clf.classes_[0]})"])
print(pd.DataFrame(clf.coef_, index=rows, columns=NUMERIC).round(2).T)

(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(model, ROOT / "models/stance_classifier.joblib")
print(model)