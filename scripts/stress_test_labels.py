"""
stress_test_labels.py — TakeMeter label stress-tester
Generates boundary cases between label pairs to pressure-test definitions.

Setup:
    pip install groq

Usage:
    export GROQ_API_KEY=your_key_here
    python scripts/stress_test_labels.py
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

LABEL_DEFINITIONS = """
analysis: The post makes a structured argument grounded in specific, verifiable evidence —
statistics, historical comparisons, tactical observations, or contextual factors. The evidence
drives the reasoning, not just decorates it.

hot_take: A bold, confident opinion stated without supporting evidence. A stat or name may
appear, but it functions as decoration rather than reasoning. Framing is confident, dismissive,
or hyperbolic.

reaction: An immediate emotional or social response to something that just happened. No claim
is being made — the post is expressing a feeling or registering a moment.
"""

BOUNDARY_PAIRS = [
    ("analysis", "hot_take"),
    ("hot_take", "reaction"),
]


def generate_boundary_cases(label_a: str, label_b: str, n: int = 8) -> list:
    prompt = f"""You are helping stress-test label definitions for a cricket discourse classifier.

Label definitions:
{LABEL_DEFINITIONS}

Generate {n} real-sounding Reddit comments from cricket subreddits (r/Cricket or r/IndianCricket)
that sit at the genuine boundary between '{label_a}' and '{label_b}'.

Rules:
- Each comment should be plausibly labelable as EITHER {label_a} or {label_b}
- Comments should sound like real cricket fans wrote them — informal, varied length
- Do NOT label them — just generate the comments
- Vary the topics: player selection, match performance, batting/bowling technique, team strategy
- Output ONLY a JSON array of strings, no other text

Example format:
["comment 1 text", "comment 2 text", ...]
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.7,
    )

    raw = response.choices[0].message.content.strip()
    try:
        comments = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract quoted lines if JSON parse fails
        comments = [line.strip().strip('",') for line in raw.split("\n")
                    if line.strip().startswith('"')]

    return comments


def main():
    print("TakeMeter — Label Stress Test")
    print("=" * 50)
    print("\nFor each comment below, decide: which label would you assign, and why?")
    print("If any stump you, update your decision rules in planning.md before annotating.\n")

    all_cases = {}

    for label_a, label_b in BOUNDARY_PAIRS:
        print(f"\n── Boundary: {label_a.upper()} ↔ {label_b.upper()} ──\n")
        cases = generate_boundary_cases(label_a, label_b)
        all_cases[f"{label_a}_vs_{label_b}"] = cases

        for i, comment in enumerate(cases, 1):
            print(f"[{i}] {comment}")
            print()

    os.makedirs("data", exist_ok=True)
    with open("data/stress_test_cases.json", "w") as f:
        json.dump(all_cases, f, indent=2)

    print("\n── Saved to data/stress_test_cases.json ──")
    print("\nNext: for any comment you struggled to label, update your decision rules in planning.md")
    print("before you start annotating 200 real examples.")


if __name__ == "__main__":
    main()
