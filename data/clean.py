import os
import pandas as pd

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())

def stance_breakdown(df, out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)

    lab = df[df["personal_label"].isin(["agreement", "disagreement"])]

    def summarize(g):
        labeled_pct_agree = (g["personal_label"] == "agreement").mean() * 100
        labeled_pct_disagree = (g["personal_label"] == "disagreement").mean() * 100
        predicted_pct_agree = (g["predicted_stance"] == "agreement").mean() * 100
        predicted_pct_disagree = (g["predicted_stance"] == "disagreement").mean() * 100

        return pd.Series({
            "n_replies": len(g),
            "labeled_pct_agree": labeled_pct_agree,
            "labeled_pct_disagree": labeled_pct_disagree,
            "predicted_pct_agree": predicted_pct_agree,
            "predicted_pct_disagree": predicted_pct_disagree,
            "labeled_difference": labeled_pct_agree - labeled_pct_disagree,
            "predicted_difference": predicted_pct_agree - predicted_pct_disagree,
        })

    by_thread = lab.groupby("response_to")[["personal_label", "predicted_stance"]].apply(summarize)
    total = summarize(lab).rename("TOTAL")

    out = pd.concat([by_thread, total.to_frame().T]).round(1)
    out.index.name = "thread"
    out.to_csv(f"{out_dir}/stance_breakdown.csv")
    return out

def process_results(df, out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)

    preds = df[["reply_id", "response_to", "personal_label", "predicted_stance"]].copy()
    preds = preds.rename(columns={"tweet_id": "reply_id"})
    preds["correct"] = preds["personal_label"] == preds["predicted_stance"]
    preds.to_csv(f"{out_dir}/predictions.csv", index=False)

    lab = preds.dropna(subset=["personal_label"])
    y, p = lab["personal_label"], lab["predicted_stance"]

    rows = [
        ("overall", "all", "n_replies", len(lab)),
        ("overall", "all", "accuracy", accuracy_score(y, p)),
        ("overall", "all", "balanced_accuracy", balanced_accuracy_score(y, p)),
        ("overall", "all", "macro_f1", f1_score(y, p, average="macro")),
        ("overall", "all", "pct_labeled_agreement", (y == "agreement").mean()),
    ]

    for tid, g in lab.groupby("response_to"):
        rows += [
            ("thread", tid, "n_replies", len(g)),
            ("thread", tid, "accuracy", g["correct"].mean()),
            ("thread", tid, "pct_labeled_agreement", (g["personal_label"] == "agreement").mean()),
            ("thread", tid, "pct_predicted_agreement", (g["predicted_stance"] == "agreement").mean()),
        ]

    pd.DataFrame(rows, columns=["level", "group", "metric", "value"]).to_csv(
        f"{out_dir}/stats.csv", index=False
    )