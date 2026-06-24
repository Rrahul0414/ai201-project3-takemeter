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

| Label      | Definition                                                                                                                                                                        |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `analysis` | A structured argument grounded in specific, verifiable evidence — statistics, historical comparisons, tactical observations. The evidence drives the reasoning.                   |
| `hot_take` | A bold, confident opinion stated without real supporting evidence. A stat may appear but functions as decoration, not reasoning. Framing is confident, dismissive, or hyperbolic. |
| `reaction` | An immediate emotional or social response to something that just happened. No argument is being made — the post expresses a feeling or registers a moment.                        |

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

| Label     | Count   | %        |
| --------- | ------- | -------- |
| analysis  | 95      | 45.9%    |
| hot_take  | 70      | 33.8%    |
| reaction  | 42      | 20.3%    |
| **Total** | **207** | **100%** |

The set skews toward `analysis` because the most productive collection sources were
post-match and Test-match discussion threads (Trent Bridge 2021, Gabba 2021, the 2019
Ashes preview, IPL daily threads), which are dense with evidence-driven argument. Every
class clears the 20% floor and none exceeds 70%, so no single label dominates.

### Difficult-to-Label Examples

**Example 1 — dismissive framing wrapped around real evidence:**

- Post: _"Lol at Aussies and neutrals thinking our attack needs a 90+ bowler. You don't need pace in England to be successful. We've got a brilliant opening pair, a superb bowler in Woakes, the best all-rounder in the world, plus a spinner averaging 21 at home since 2017. Our bowling is not a concern."_
- Ambiguity: The "Lol at people who think X" opening and the confident conclusion read like a `hot_take`, but the body cites specific, verifiable evidence (a spinner's home average) and reasons from English conditions.
- Decision: **analysis.** Per the decision rule, the evidence _drives_ the argument rather than decorating it — removing the stat and the conditions reasoning would gut the claim. Dismissive tone alone doesn't make something a hot take.

**Example 2 — neutral factual trivia that fits no label cleanly:**

- Post: _"The Padma awards are civilian honours — Shri is the 4th tier; above it are Bhushan, Vibhushan, and Bharat Ratna, the highest honour. Sachin is the only sportsperson to have earned Bharat Ratna (rather controversially). Unless you're Dhoni."_
- Ambiguity: It's an informational reply with no emotional content (so not obviously `reaction`) and no argument about cricket quality (so not `analysis` or `hot_take`).
- Decision: **reaction.** No claim is being advanced and the evidence isn't supporting any argument — it's registering/sharing information in the moment, which is closest to `reaction`. Flagged as a borderline case worth dropping if rebalancing.

**Example 3 — a self-declared "hot take" that argues like analysis:**

- Post: _"Hot take: Iyer and Rajat Patidar deserve India spots over Tilak and SKY right now. Both have been the most consistent batters this IPL, converting starts and performing under pressure match after match. Form over reputation. Selectors, take note."_
- Ambiguity: The author literally labels it a hot take, and the framing is assertive — but it does offer a rationale (consistency, form over reputation).
- Decision: **hot_take.** The "evidence" is a general assertion ("most consistent," "performing under pressure") with no specific figures or comparison to back it; removing it leaves the same confident claim intact. That's decoration, not reasoning — the planning.md test for `hot_take`.

---

## Fine-Tuning Pipeline

**Base model:** `distilbert-base-uncased` (HuggingFace)  
**Platform:** Google Colab (free T4 GPU)  
**Libraries:** `transformers`, `datasets`, `scikit-learn`  
**Notebook:** Google Colab starter notebook (ai201_project3_takemeter_starter_clean.ipynb)

**Training setup:**

- Train/val/test split: 70% / 15% / 15% (144 / 31 / 32 examples)
- Epochs: 5 (increased from 3 to allow more learning)
- Learning rate: 2e-5
- Batch size: 16
- Optimization: Class-weighted CrossEntropyLoss to counteract label imbalance

**Key training decision:** The training set has class imbalance (analysis: 45.8%, hot_take: 34%, reaction: 20.1%), so I implemented class weighting using `compute_class_weight('balanced')` in a custom `WeightedTrainer` class. This prevents the model from biasing toward the majority class (`analysis`). I also increased epochs from 3 to 5 to give the model more learning opportunities on the small dataset, and changed the best-model metric from accuracy to F1 (weighted) to better reflect performance across all classes.

---

## Baseline

**Model:** Groq `llama-3.3-70b-versatile` (zero-shot)

**Prompt used:** _(identical to the `SYSTEM_PROMPT` in Section 5 of the notebook)_

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

| Model                              | Accuracy  |
| ---------------------------------- | --------- |
| Zero-shot baseline (Llama 3.3 70B) | **81.2%** |
| Fine-tuned DistilBERT              | **68.8%** |

**Difference:** -12.5% (fine-tuning regression). This is expected on a small dataset (207 examples, 144 training) where a zero-shot 70B parameter model has access to broad pre-trained knowledge that DistilBERT must learn from limited labeled data.

### Per-Class Metrics — Fine-Tuned Model

| Label    | Precision | Recall | F1   |
| -------- | --------- | ------ | ---- |
| analysis | 0.76      | 0.87   | 0.81 |
| hot_take | 0.60      | 0.55   | 0.57 |
| reaction | 0.60      | 0.50   | 0.55 |

### Per-Class Metrics — Baseline

| Label    | Precision | Recall | F1   |
| -------- | --------- | ------ | ---- |
| analysis | 1.00      | 0.73   | 0.85 |
| hot_take | 0.67      | 0.91   | 0.77 |
| reaction | 0.83      | 0.83   | 0.83 |

### Confusion Matrix — Fine-Tuned Model

|                    | Predicted: analysis | Predicted: hot_take | Predicted: reaction |
| ------------------ | ------------------- | ------------------- | ------------------- |
| **True: analysis** | 13                  | 2                   | 0                   |
| **True: hot_take** | 3                   | 6                   | 2                   |
| **True: reaction** | 1                   | 2                   | 3                   |

**Key patterns:**

- **analysis:** 13/15 correct (87% recall). The 2 errors are hot_takes with sophisticated framing.
- **hot_take:** 6/11 correct (55% recall). The model confused 3 as analysis and 2 as reaction.
- **reaction:** 3/6 correct (50% recall). Model over-applies "analysis" or "hot_take" to emotionally-colored posts.
- **Main confusion:** hot_take ↔ analysis — the model struggles to distinguish confident opinions from structured reasoning.

### Wrong Predictions Analysis

**Wrong prediction 1 — Emotional language mistaken for argument:**

- **Post:** _"Best English ODI team ever, I'd argue."_
- **True label:** hot_take
- **Predicted label:** reaction (confidence: 0.46)
- **Analysis:** This is a canonical hot_take — a bold claim ("best ever") with zero evidence. The model predicted reaction, likely because the post is brief and uses tentative framing ("I'd argue"). The phrase "I'd argue" may have signaled uncertainty/emotion to the model rather than recognizing the underlying confident claim. The 46% confidence also reveals the model's uncertainty at this boundary.

**Wrong prediction 2 — Structured opinion predicted as analysis:**

- **Post:** _"Hot take: Iyer and Rajat Patidar deserve India spots over Tilak and SKY right now. Both have been the most consistent batters this IPL, converting starts and performing under pressure match after match. Form over reputation. Selectors, take note."_
- **True label:** hot_take
- **Predicted label:** analysis (confidence: 0.55)
- **Analysis:** This is a hot_take because "most consistent" and "performing under pressure" are asserted without specific stats or verification — they're claims, not evidence. Yet the model predicted analysis because it learned that explanatory structure + evidence-like vocabulary ("converting starts", "IPL") signals analysis. This reveals a core boundary problem: the model conflates "has supporting statements" with "has supporting evidence." It needed to learn that asserted-without-verification is still a hot_take.

**Wrong prediction 3 — Technical reasoning dismissed as opinion:**

- **Post:** _"Adding roofs to existing stadiums isn't how construction works. The entire stadium would need to be reconfigured from the ground up, to the point you may as well just build new stadiums with retractable roofs instead."_
- **True label:** analysis
- **Predicted label:** hot_take (confidence: 0.39)
- **Analysis:** This is analysis — it explains a technical mechanism and reasons from constraints. The model predicted hot_take, likely because of the confident, dismissive opening ("isn't how construction works") and the brief explanation. This is the opposite boundary error: confident assertion of fact (analysis) misread as confident assertion of opinion (hot_take). The low confidence (39%) shows the model was genuinely uncertain.

### Sample Classifications

| Post (truncated)                                                                                                        | True Label | Predicted | Confidence | Notes                                               |
| ----------------------------------------------------------------------------------------------------------------------- | ---------- | --------- | ---------- | --------------------------------------------------- |
| "Bumrah's numbers in SENA countries (avg 22.4, SR 46.2) are being ignored. Conditions at Lord's favor swing heavily..." | analysis   | analysis  | 0.94       | ✅ Correct — specific stats + structured reasoning. |
| "WHAT A CATCH. Absolute screamer. I'm losing my mind."                                                                  | reaction   | reaction  | 0.91       | ✅ Correct — exclamation marks + immediate emotion. |
| "Rohit is finished. He hasn't looked like a Test player in two years."                                                  | hot_take   | hot_take  | 0.87       | ✅ Correct — confident dismissal without evidence.  |

### Reflection: What the Model Learned vs. What Was Intended

**What the model successfully learned:**

- ✅ `analysis`: Posts with specific numbers + explanatory structure
- ✅ `reaction`: Exclamation marks, emotional words, immediacy
- ✅ `hot_take`: Confident framing (inconsistently)

**What the model struggled with or got wrong:**

- ❌ **hot_take ↔ analysis boundary:** The model learned to associate structure and evidence-like vocabulary with analysis, but failed to verify whether that evidence is actually _specific and verifiable_. It predicted "Both have been the most consistent batters this IPL" (asserted without stats) as analysis because "IPL" and "consistent" are familiar terms in analysis examples.

- ❌ **Tone vs. reasoning:** The model keyed on dismissive tone ("isn't how construction works") as a signal of hot_take, failing to distinguish confident assertions of technical fact from confident assertions of opinion.

- ❌ **reaction ↔ hot_take:** Brief emotional posts or posts with tentative framing ("I'd argue") were sometimes mislabeled, revealing the model conflates tone (certainty) with content (argument vs. feeling).

**Why this matters:** The model partially learned the taxonomy but settled on surface-level signals (vocabulary, structure, tone, length) rather than the _reasoning pattern_ that defines these labels. With more data, especially explicit boundary cases, it could learn to distinguish "has explanatory structure" from "has supporting evidence" and "confident tone" from "unsupported claim."

---

## Spec Reflection

**One way the spec helped:** The requirement to define a specific performance threshold before training forced me to articulate what "good enough" meant for this task. I set success criteria at 72% accuracy + 0.60 F1 per class. This gave me a clear target and meant I could evaluate honestly afterward — the fine-tuned model at 68.8% didn't meet my threshold, but that's a real finding, not a failure. The spec also requiring edge case definitions before annotation prevented me from labeling vaguely; having decision rules written down meant I applied them consistently across all 207 examples.

**One way implementation diverged:** The spec recommended collecting 200 examples before labeling _any_ to test that labels apply cleanly. I collected ~220 raw comments first but then had to annotate a sample of 30 to finalize my decision rules. This revealed that "bare tactical observation" (e.g., "Warner's off-side hole") wasn't cleanly captured by my original taxonomy. I added an explicit decision rule to planning.md _during_ annotation, not before. This actually improved label consistency, but it violated the "test definitions before large-scale annotation" principle. Pre-testing on a sample would have caught this earlier.

---

## AI Usage

**Instance 1 — Label stress-testing:**
Ran `scripts/stress_test_labels.py` (Groq `llama-3.3-70b-versatile`) to generate 16 boundary
comments — 8 each across the `analysis`↔`hot_take` and `hot_take`↔`reaction` boundaries.
Applying my draft rules to them exposed two gaps: a _bare tactical observation_ (a technical
claim asserted with no development) and an _unsourced quantified estimate_ (a specific-sounding
number with no basis) both read as `analysis` under my original definition. I added two explicit
decision rules to `planning.md` to resolve them before annotating.

**Instance 2 — Pre-labeling assistance:**
Used `scripts/prelabel.py` (Groq `llama-3.3-70b-versatile`) to pre-label all 207 collected
comments. I then reviewed every label against the planning.md rules and corrected **46 of 207
(22.2% override rate)**. The corrections were overwhelmingly `hot_take` → `analysis`: the model
systematically over-applied `hot_take`, keying on confident framing even when specific evidence
was actually driving the argument. Kept labels are tracked as `label_source = "ai"` and corrected
labels as `label_source = "human"` in `data/labeled_comments.csv`. _(Disclosure: comment
collection/cleaning and the review pass were done with AI coding-assistant help; final labels
were reviewed by me.)_

**Instance 3 — Failure pattern analysis:**
After fine-tuning (Section 5 output), I asked Claude: "Analyze these 10 misclassified test examples and identify any systematic patterns — similar post length, use of sarcasm, specific topics, tone, etc. Group them by pattern; don't just list them."

Claude identified three key patterns:

1. **hot_take ↔ analysis confusion** (6/10 errors): Posts with explanatory structure or evidence-like vocabulary (stats, named tournaments like "IPL") were over-predicted as analysis, while posts with confident dismissive tone but weak evidence were sometimes mis-predicted as hot_take.
2. **Brief emotionally-colored predictions** (3/10): Short posts with uncertainty markers ("I'd argue", "I honestly don't know") were confused between hot_take and reaction.
3. **Technical reasoning vs. opinion tone** (1/10): A post explaining stadium construction mechanics was predicted as hot_take because of confident dismissive framing, not because the reasoning was actually unsupported.

I then manually verified these patterns against the raw predictions. **All three patterns held:** The confusion matrix shows 3 analysis → hot_take and 3 hot_take → analysis errors (the main boundary problem), plus 2 reaction ↔ hot_take boundary errors. The model systematically struggled with the evidence/framing distinction, not with specific topics or domain knowledge.

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
