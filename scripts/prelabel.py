"""
prelabel.py — TakeMeter LLM-assisted annotation
Pre-labels raw comments using Groq's llama-3.3-70b-versatile, for human review and correction.

Setup:
    pip install groq pandas

Usage:
    export GROQ_API_KEY=your_key_here
    python scripts/prelabel.py --input data/raw_comments.csv --output data/prelabeled.csv

After running:
    Open prelabeled.csv and review every row.
    The 'label' column has the AI suggestion.
    The 'label_source' column tracks 'ai' vs 'human'.
    Change 'label' if you disagree — set 'label_source' to 'human' for overrides.
    Your final labeled_comments.csv should include the label_source column for disclosure.
"""

import os
import argparse
import time
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a cricket discourse classifier. Your job is to assign exactly one label
to each Reddit comment from cricket subreddits.

Labels:
- analysis: Structured argument grounded in specific verifiable evidence (stats, historical
  comparisons, tactical observations). Evidence drives the reasoning, not just decorates it.
- hot_take: Bold confident opinion without real supporting evidence. May cite a stat but uses
  it decoratively. Confident, dismissive, or hyperbolic framing.
- reaction: Immediate emotional response to something that just happened. No argument being
  made — just expressing a feeling or registering a moment.

Decision rules for hard cases:
- If a post cites a stat but makes no structured argument (the stat is there for effect, not
  reasoning), label it hot_take not analysis.
- If a post is emotional but also makes a claim (even loosely), label it hot_take not reaction.
  Reaction is reserved for purely expressive posts with no argumentative intent.

Output ONLY the label — one of: analysis, hot_take, reaction
No explanation, no punctuation, no other text."""


def classify_comment(text: str, retries: int = 2) -> str:
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=10,
                temperature=0,
            )
            label = response.choices[0].message.content.strip().lower()
            # Normalize "hot take" → "hot_take": the model frequently emits a
            # space where our label uses an underscore.
            label = label.replace(" ", "_")
            if label in ("analysis", "hot_take", "reaction"):
                return label
            for valid in ("analysis", "hot_take", "reaction"):
                if valid in label:
                    return valid
            return "NEEDS_REVIEW"
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
            else:
                print(f"  Failed after {retries+1} attempts: {e}")
                return "NEEDS_REVIEW"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw_comments.csv")
    parser.add_argument("--output", default="data/prelabeled.csv")
    parser.add_argument("--batch", type=int, default=20,
                        help="Print progress every N rows")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} comments from {args.input}")

    # When the scraper writes empty labels, pandas reads the column as all-NaN
    # (float dtype), which breaks the .str accessor. Coerce to string first.
    if "label" not in df.columns:
        df["label"] = ""
    df["label"] = df["label"].fillna("").astype(str)
    unlabeled_mask = df["label"].str.strip() == ""
    unlabeled_df = df[unlabeled_mask].copy()
    print(f"{len(unlabeled_df)} unlabeled rows to process")

    labels = []
    for i, (_, row) in enumerate(unlabeled_df.iterrows()):
        label = classify_comment(row["text"])
        labels.append(label)
        if (i + 1) % args.batch == 0:
            print(f"  Processed {i+1}/{len(unlabeled_df)}...")
        time.sleep(0.2)  # stay within Groq free tier rate limits

    unlabeled_df["label"] = labels
    unlabeled_df["label_source"] = "ai"

    df.loc[unlabeled_mask, "label"] = labels
    df.loc[unlabeled_mask, "label_source"] = "ai"
    df["label_source"] = df["label_source"].fillna("human")

    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")

    print("\n── AI Pre-label Distribution ──")
    print(df["label"].value_counts().to_string())
    needs_review = (df["label"] == "NEEDS_REVIEW").sum()
    if needs_review:
        print(f"\n⚠  {needs_review} rows marked NEEDS_REVIEW — assign these manually")

    print("\n── Next steps ──")
    print(f"1. Open {args.output} in a spreadsheet")
    print("2. Review EVERY row — read the text and check the AI label")
    print("3. Change 'label' for any you disagree with; set 'label_source' to 'human'")
    print("4. Track your override rate: if >30%, tighten your definitions in planning.md")
    print("5. Save final version as data/labeled_comments.csv")


if __name__ == "__main__":
    main()
