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

**Label distribution:**

| Label | Count | % |
|-------|-------|---|
| analysis | — | — |
| hot_take | — | — |
| reaction | — | — |
| **Total** | **—** | **100%** |

*(Fill in after annotation is complete)*

### Difficult-to-Label Examples

**Example 1:** *(To be filled during annotation)*
- Post: 
- Ambiguity: 
- Decision: 

**Example 2:** *(To be filled during annotation)*
- Post: 
- Ambiguity: 
- Decision: 

**Example 3:** *(To be filled during annotation)*
- Post: 
- Ambiguity: 
- Decision: 

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
Ran `scripts/stress_test_labels.py` which prompted Claude to generate 8 boundary cases between
`analysis` and `hot_take`. Several cases exposed ambiguity in my original definition: posts
that cited a stat but used it for rhetorical effect. This led me to add the explicit decision
rule in `planning.md`: "if removing the stat would leave a claim that still feels confident and
complete, label it hot_take." I discarded 2 of the 8 generated cases as unrealistic
(they were too clearly one label).

**Instance 2 — Pre-labeling assistance:**
Used `scripts/prelabel.py` to pre-label batches of 20 comments at a time using Claude.
Reviewed and corrected every AI-assigned label individually. Override rate: *(fill in after annotation)*.
All pre-labeled rows are tracked with `label_source = "ai"` in the dataset.

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
