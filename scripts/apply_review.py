"""
apply_review.py — records a rule-based review pass over the AI pre-labels.

Each entry in CHANGES is a row index (0-based, matching prelabeled.csv order)
whose AI label was overridden during review, applying the planning.md decision
rules — chiefly correcting the model's systematic over-use of `hot_take` for
comments whose evidence actually DRIVES the reasoning (→ analysis), and a few
expressive/quip rows the model over-read as takes (→ reaction), plus genuine
claims it under-read as reactions (→ hot_take).

label_source:
  'ai'           = AI pre-label kept on review
  'ai-corrected' = AI pre-label changed on review (this is an AI-assisted pass)
The human annotator should flip rows they personally verify/own to 'human'.
"""
import pandas as pd

# index -> corrected label
CHANGES = {
    # hot_take -> analysis (evidence/tactical reasoning drives the comment)
    23: "analysis", 38: "analysis", 47: "analysis", 54: "analysis", 55: "analysis",
    60: "analysis", 66: "analysis", 67: "analysis", 78: "analysis", 80: "analysis",
    90: "analysis", 96: "analysis", 100: "analysis", 102: "analysis", 116: "analysis",
    125: "analysis", 127: "analysis", 129: "analysis", 132: "analysis", 136: "analysis",
    152: "analysis", 155: "analysis", 162: "analysis", 164: "analysis", 169: "analysis",
    171: "analysis", 172: "analysis", 174: "analysis", 175: "analysis", 186: "analysis",
    189: "analysis", 192: "analysis", 193: "analysis", 204: "analysis",
    # hot_take -> reaction (quip / expressive, no argument advanced)
    5: "reaction", 179: "reaction", 195: "reaction",
    # reaction -> hot_take (a real claim is being advanced)
    7: "hot_take", 11: "hot_take", 12: "hot_take", 30: "hot_take", 52: "hot_take",
    183: "hot_take",
    # reaction -> analysis (specific evidence / technical observation)
    25: "analysis", 37: "analysis", 121: "analysis",
}

df = pd.read_csv("data/prelabeled.csv")
df = df.reset_index(drop=True)

ai_label = df["label"].copy()
for idx, new_label in CHANGES.items():
    df.at[idx, "label"] = new_label

df["label_source"] = ["ai-corrected" if i in CHANGES else "ai" for i in range(len(df))]

# Final column order matches the labeled_comments.csv header
cols = ["text", "subreddit", "thread_title", "comment_score", "label", "label_source", "notes"]
df = df[cols]
df.to_csv("data/labeled_comments.csv", index=False)

n = len(df)
print(f"Wrote data/labeled_comments.csv ({n} rows)")
print(f"Overrides: {len(CHANGES)} ({len(CHANGES)/n:.1%} override rate)\n")
print("Final label distribution:")
print(df["label"].value_counts().to_string())
print()
print("As % of dataset:")
print((df["label"].value_counts(normalize=True) * 100).round(1).to_string())
print()
print("label_source:")
print(df["label_source"].value_counts().to_string())
