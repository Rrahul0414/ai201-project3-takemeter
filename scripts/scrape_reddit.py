"""
scrape_reddit.py — TakeMeter data collection script
Pulls comments from r/Cricket and r/IndianCricket for labeling.

Setup:
    pip install praw pandas

Usage:
    1. Create a Reddit app at https://www.reddit.com/prefs/apps
       (choose "script" type, redirect URI: http://localhost:8080)
    2. Fill in your credentials in the REDDIT_CREDENTIALS block below
    3. Run: python scripts/scrape_reddit.py

Output:
    data/raw_comments.csv — unlabeled comments ready for annotation
"""

import praw
import pandas as pd
import re
import time
from datetime import datetime

# ── Credentials ──────────────────────────────────────────────────────────────
# Fill these in. Never commit real credentials to GitHub.
REDDIT_CREDENTIALS = {
    "client_id": "YOUR_CLIENT_ID",
    "client_secret": "YOUR_CLIENT_SECRET",
    "user_agent": "takemeter-scraper/1.0 by YOUR_REDDIT_USERNAME",
}

# ── Config ────────────────────────────────────────────────────────────────────
SUBREDDITS = ["Cricket", "IndianCricket"]

# Thread types most likely to yield all three label classes:
#   - match threads      → high reaction + hot_take
#   - selection threads  → high hot_take + some analysis
#   - discussion/deep    → high analysis
SEARCH_QUERIES = [
    "match thread",
    "selection",
    "test squad",
    "playing XI",
    "post match",
    "discussion",
    "analysis",
]

MIN_CHARS = 30       # shorter than this is noise (one-word replies, emoji only)
MAX_CHARS = 500      # longer than this tends to be walls-of-text or pastas
MIN_SCORE = 2        # filter out completely downvoted/ignored comments
TARGET_TOTAL = 300   # collect more than 200 so we have room to balance labels

# ─────────────────────────────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """Strip links, Reddit formatting artifacts, and excessive whitespace."""
    text = re.sub(r"http\S+", "", text)           # URLs
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)    # markdown links
    text = re.sub(r"[>\*\_\~\^]", "", text)       # markdown symbols
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_usable(text: str) -> bool:
    """Basic quality filter."""
    if len(text) < MIN_CHARS or len(text) > MAX_CHARS:
        return False
    # Skip obvious bots / AutoModerator
    lower = text.lower()
    if any(kw in lower for kw in ["i am a bot", "automoderator", "join our discord"]):
        return False
    return True


def scrape(limit_per_query: int = 5) -> pd.DataFrame:
    reddit = praw.Reddit(**REDDIT_CREDENTIALS)
    seen = set()
    rows = []

    for subreddit_name in SUBREDDITS:
        subreddit = reddit.subreddit(subreddit_name)
        print(f"\n── Scraping r/{subreddit_name} ──")

        for query in SEARCH_QUERIES:
            print(f"  Searching: '{query}'")
            try:
                results = subreddit.search(query, sort="top", time_filter="year", limit=limit_per_query)
                for submission in results:
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments.list():
                        if comment.score < MIN_SCORE:
                            continue
                        text = clean_text(comment.body)
                        if not is_usable(text):
                            continue
                        if text in seen:
                            continue
                        seen.add(text)
                        rows.append({
                            "text": text,
                            "subreddit": subreddit_name,
                            "thread_title": submission.title[:100],
                            "comment_score": comment.score,
                            "label": "",      # to be filled during annotation
                            "notes": "",      # annotation notes for hard cases
                        })
                        if len(rows) >= TARGET_TOTAL:
                            break
                    if len(rows) >= TARGET_TOTAL:
                        break
                time.sleep(1)  # be polite to the API
            except Exception as e:
                print(f"    Warning: {e}")
                continue

            if len(rows) >= TARGET_TOTAL:
                break

        if len(rows) >= TARGET_TOTAL:
            break

    df = pd.DataFrame(rows)
    print(f"\nCollected {len(df)} usable comments.")
    return df


def main():
    df = scrape(limit_per_query=8)

    out_path = "data/raw_comments.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    # Print label distribution reminder
    print("\n── Next steps ──")
    print("1. Open data/raw_comments.csv in a spreadsheet (Excel, Google Sheets)")
    print("2. Fill the 'label' column: analysis | hot_take | reaction")
    print("3. Use the 'notes' column for ambiguous cases")
    print("4. Aim for: ~33% per label, no label > 70%")
    print("5. Save final labeled file as data/labeled_comments.csv")


if __name__ == "__main__":
    main()
