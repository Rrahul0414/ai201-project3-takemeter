# TakeMeter — Cricket Discourse Classifier

A fine-tuned text classifier that evaluates the quality of cricket fan commentary on Reddit,
distinguishing between structured analysis, hot takes, and in-the-moment reactions.

---

## Community

**r/Cricket** and **r/IndianCricket** — two of the largest cricket communities on Reddit,
with active commentary on matches, player selection, tactics, and cricket culture.

Cricket subreddits are a strong fit for this task because discourse quality varies enormously
and visibly. A single match thread contains tactical breakdowns citing bowling strike rates
alongside pure-hype one-liners. The community also has a strong statistical culture — fans
regularly cite averages, conversion ratios, economy rates — which makes the
analysis/hot-take boundary genuinely legible rather than subjectively vague. The distinctions
this classifier captures reflect norms the community itself already holds.

---

## Label Taxonomy

| Label | Definition |
|-------|-----------|
| `analysis` | A structured argument grounded in specific, verifiable evidence — statistics, historical comparisons, tactical observations. The evidence drives the reasoning. |
| `hot_take` | A bold, confident opinion stated without real supporting evidence. A stat may appear but functions as decoration, not reasoning. Framing is confident, dismissive, or hyperbolic. |
| `reaction` | An immediate emotional or social response to something that just happened. No argument is being made — the post expresses a feeling or registers a moment. |

### Example Posts

**analysis:**
> "Bumrah's numbers in SENA countries (avg 22.4, SR 46.2 over 18 tests) are being ignored here. The conditions at Lord's favor swing heavily, and he's the only Indian pacer who's been consistently effective there. Dropping him would be the wrong call."

> "Kohli's conversion rate (50s to 100s) has dropped from 47% pre-2020 to 31% since. That's not a slump — that's a structural change in how he's batting in the 70–90 range. The feet aren't moving the same way against short-pitched bowling."

**hot_take:**
> "Rohit is finished. He hasn't looked like a Test player in two years. India needs to move on before the Champions Trophy."

> "England's Bazball is a fraud tactic that will never work in the subcontinent. Anyone who thinks otherwise doesn't understand real cricket."

**reaction:**
> "WHAT A CATCH. Absolute screamer. I'm losing my mind."

> "Can't believe they dropped Jadeja for this. Heartbroken."

---

## Data Collection

**Source:** r/Cricket and r/IndianCricket via Reddit API (PRAW)  
**Script:** `scripts/scrape_reddit.py`  
**Thread types targeted:** match threads, selection announcement threads, post-match discussions

**Labeling process:** Comments were collected, cleaned (URLs removed, markdown stripped,
filtered to 30–500 characters), and then pre-labeled using Claude (`scripts/prelabel.py`)
with the full label definitions and decision rules from `planning.md`. Every AI pre-label
was reviewed and corrected individually. The final labels reflect human judgment.

**Label distribution:** (207 labeled comments)

| Label | Count | % |
|-------|-------|---|
| analysis | 95 | 45.9% |
| hot_take | 70 | 33.8% |
| reaction | 42 | 20.3% |
| **Total** | **207** | **100%** |

The set skews toward `analysis` because the most productive collection sources were
post-match and Test-match discussion threads (Trent Bridge 2021, Gabba 2021, the 2019
Ashes preview, IPL daily threads), which are dense with evidence-driven argument. Every
class clears the 20% floor and none exceeds 70%, so no single label dominates.

### Difficult-to-Label Examples

**Example 1 — dismissive framing wrapped around real evidence:**
- Post: *"Lol at Aussies and neutrals thinking our attack needs a 90+ bowler. You don't need pace in England to be successful. We've got a brilliant opening pair, a superb bowler in Woakes, the best all-rounder in the world, plus a spinner averaging 21 at home since 2017. Our bowling is not a concern."*
- Ambiguity: The "Lol at people who think X" opening and the confident conclusion read like a `hot_take`, but the body cites specific, verifiable evidence (a spinner's home average) and reasons from English conditions.
- Decision: **analysis.** Per the decision rule, the evidence *drives* the argument rather than decorating it — removing the stat and the conditions reasoning would gut the claim. Dismissive tone alone doesn't make something a hot take.

**Example 2 — neutral factual trivia that fits no label cleanly:**
- Post: *"The Padma awards are civilian honours — Shri is the 4th tier; above it are Bhushan, Vibhushan, and Bharat Ratna, the highest honour. Sachin is the only sportsperson to have earned Bharat Ratna (rather controversially). Unless you're Dhoni."*
- Ambiguity: It's an informational reply with no emotional content (so not obviously `reaction`) and no argument about cricket quality (so not `analysis` or `hot_take`).
- Decision: **reaction.** No claim is being advanced and the evidence isn't supporting any argument — it's registering/sharing information in the moment, which is closest to `reaction`. Flagged as a borderline case worth dropping if rebalancing.

**Example 3 — a self-declared "hot take" that argues like analysis:**
- Post: *"Hot take: Iyer and Rajat Patidar deserve India spots over Tilak and SKY right now. Both have been the most consistent batters this IPL, converting starts and performing under pressure match after match. Form over reputation. Selectors, take note."*
- Ambiguity: The author literally labels it a hot take, and the framing is assertive — but it does offer a rationale (consistency, form over reputation).
- Decision: **hot_take.** The "evidence" is a general assertion ("most consistent," "performing under pressure") with no specific figures or comparison to back it; removing it leaves the same confident claim intact. That's decoration, not reasoning — the planning.md test for `hot_take`.

---

## Fine-Tuning Pipeline

**Base model:** `distilbert-base-uncased` (HuggingFace)  
**Platform:** Google Colab (free T4 GPU)  
**Libraries:** `transformers`, `datasets`, `scikit-learn`  
**Notebook:** [TakeMeter Colab Notebook](#) *(add link after copying)*

**Training setup:**
- Train/val/test split: 70% / 15% / 15% (handled by notebook)
- Epochs: 3
- Learning rate: 2e-5
- Batch size: 16

**Key training decision:** *(Fill in after training — e.g., "I increased epochs from 3 to 5
because validation loss was still decreasing at epoch 3. The additional epochs improved
per-class F1 on hot_take from 0.58 to 0.67 without signs of overfitting on analysis or reaction.")*

---

## Baseline

**Model:** Groq `llama-3.3-70b-versatile` (zero-shot)

**Prompt used:** *(identical to the `SYSTEM_PROMPT` in Section 5 of the notebook)*
```
You are a cricket discourse classifier for Reddit comments from r/Cricket and r/IndianCricket.
Assign each comment to exactly one of the following categories.

analysis: A structured argument grounded in specific, verifiable evidence — statistics, historical
comparisons, or tactical observations. The evidence drives the reasoning, not just decorates it.
Example: "Bumrah's numbers in SENA countries (avg 22.4, SR 46.2 over 18 tests) are being ignored
here. Conditions at Lord's favor swing heavily, and he's the only Indian pacer who's been
consistently effective there. Dropping him would be the wrong call."

hot_take: A bold, confident opinion stated without real supporting evidence. A stat may appear but
functions as decoration rather than reasoning. The framing is confident, dismissive, or hyperbolic.
Example: "Rohit is finished. He hasn't looked like a Test player in two years. India needs to move
on before the Champions Trophy."

reaction: An immediate emotional or social response to something that just happened. No argument is
being made — the comment expresses a feeling or registers a moment.
Example: "WHAT A CATCH. Absolute screamer. I'm losing my mind."

Decision rules for hard cases:
- If a comment cites a stat but makes no structured argument (the stat is there for effect, not
  reasoning), label it hot_take, not analysis.
- If a comment is emotional but also advances a claim (however loosely), label it hot_take, not
  reaction. Reaction is reserved for purely expressive comments.

Respond with ONLY the label name: analysis, hot_take, or reaction.
Do not explain your reasoning.

Valid labels:
analysis
hot_take
reaction
```

**How results were collected:** Each test example was passed to the Groq API via the starter
notebook's Section 5. Responses were parsed and matched against true labels. Unparseable
responses were counted as incorrect.

---

## Evaluation Report

### Overall Accuracy

| Model | Accuracy |
|-------|----------|
| Zero-shot baseline (Llama 3.3 70B) | — |
| Fine-tuned DistilBERT | — |

*(Fill in after training and baseline evaluation)*

### Per-Class Metrics — Fine-Tuned Model

| Label | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| analysis | — | — | — |
| hot_take | — | — | — |
| reaction | — | — | — |

### Per-Class Metrics — Baseline

| Label | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| analysis | — | — | — |
| hot_take | — | — | — |
| reaction | — | — | — |

### Confusion Matrix — Fine-Tuned Model

*(Replace dashes with actual counts after training)*

|  | Predicted: analysis | Predicted: hot_take | Predicted: reaction |
|--|---------------------|---------------------|---------------------|
| **True: analysis** | — | — | — |
| **True: hot_take** | — | — | — |
| **True: reaction** | — | — | — |

### Wrong Predictions Analysis

**Wrong prediction 1:**
- Post: 
- True label: 
- Predicted label: 
- Analysis: 

**Wrong prediction 2:**
- Post: 
- True label: 
- Predicted label: 
- Analysis: 

**Wrong prediction 3:**
- Post: 
- True label: 
- Predicted label: 
- Analysis: 

### Sample Classifications

*(3–5 posts run through the fine-tuned model with predicted label and confidence)*

| Post (truncated) | True Label | Predicted | Confidence | Notes |
|------------------|-----------|-----------|------------|-------|
| | | | | |
| | | | | |
| | | | | |

### Reflection: What the Model Learned vs. What Was Intended

*(Fill in after evaluation — e.g., "The model learned to associate statistical vocabulary with
`analysis` and exclamation marks with `reaction`, but it struggled with the core distinction:
whether evidence is doing argumentative work or just decorating a hot take. Posts with one
embedded stat and aggressive framing were systematically misclassified as analysis because
the model keyed on the stat rather than the framing.")*

---

## Spec Reflection

**One way the spec helped:** *(Fill in — e.g., "The instruction to define a specific performance
threshold before training forced me to articulate what 'good enough' meant for this task...")*

**One way implementation diverged:** *(Fill in — e.g., "The spec suggested collecting 200 examples
before labeling any, but I found I needed to annotate the first 30 to finalize my decision rules...")*

---

## AI Usage

**Instance 1 — Label stress-testing:**
Ran `scripts/stress_test_labels.py` (Groq `llama-3.3-70b-versatile`) to generate 16 boundary
comments — 8 each across the `analysis`↔`hot_take` and `hot_take`↔`reaction` boundaries.
Applying my draft rules to them exposed two gaps: a *bare tactical observation* (a technical
claim asserted with no development) and an *unsourced quantified estimate* (a specific-sounding
number with no basis) both read as `analysis` under my original definition. I added two explicit
decision rules to `planning.md` to resolve them before annotating.

**Instance 2 — Pre-labeling assistance:**
Used `scripts/prelabel.py` (Groq `llama-3.3-70b-versatile`) to pre-label all 207 collected
comments. I then reviewed every label against the planning.md rules and corrected **46 of 207
(22.2% override rate)**. The corrections were overwhelmingly `hot_take` → `analysis`: the model
systematically over-applied `hot_take`, keying on confident framing even when specific evidence
was actually driving the argument. Kept labels are tracked as `label_source = "ai"` and corrected
labels as `label_source = "human"` in `data/labeled_comments.csv`. *(Disclosure: comment
collection/cleaning and the review pass were done with AI coding-assistant help; final labels
were reviewed by me.)*

**Instance 3 — Failure pattern analysis:**
*(Fill in after fine-tuning — describe what you asked the AI to find, what it identified,
and which patterns you confirmed or rejected after reviewing the raw examples yourself.)*

---

## Setup

```bash
# Install dependencies (local data pipeline; fine-tuning runs on Colab)
pip install -r requirements.txt

# Set your Groq API key (copy .env.example → .env and fill it in)
cp .env.example .env   # then edit .env to add your real key

# Collect raw data
python scripts/scrape_reddit.py
# → fill in Reddit credentials in the script first

# Run label stress test (before annotating)
python scripts/stress_test_labels.py

# Pre-label for annotation assistance
python scripts/prelabel.py --input data/raw_comments.csv --output data/prelabeled.csv
# → review every row in prelabeled.csv and save final version as data/labeled_comments.csv
```

See `planning.md` for full label definitions, edge case decision rules, and evaluation reasoning.
