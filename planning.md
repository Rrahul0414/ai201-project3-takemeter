# TakeMeter — Planning Document
**Community:** r/Cricket (and r/IndianCricket)
**Task:** Discourse quality classification for cricket fan commentary

---

## 1. Community Choice

Cricket communities on Reddit — primarily r/Cricket (~700k members) and r/IndianCricket (~500k members) — are ideal for this task for three reasons.

First, discourse quality varies enormously and visibly. A single match thread will contain someone citing a batter's strike rate in SENA countries alongside someone writing "he's just trash lol." Both are real posts, and the gap between them is the signal this classifier is designed to capture.

Second, cricket has a deep statistical culture. Unlike most sports, cricket fans routinely cite averages, economy rates, conversion ratios, bowling strike rates, and head-to-head records. This makes the analysis/hot-take boundary legible — you can actually check whether a post is reasoning with evidence.

Third, the community has strong informal norms about what counts as a "quality take." Regulars recognize and upvote posts that show tactical or historical depth. This means the distinctions in this taxonomy aren't invented — they already exist in the community's own evaluations.

---

## 2. Label Taxonomy

### Labels (3 total)

**`analysis`**
The post makes a structured argument grounded in specific, verifiable evidence — statistics, historical comparisons, tactical observations, or contextual factors (pitch conditions, match situation, opposition bowling attack). The evidence is used to support a claim, not just decorate it. The reader comes away understanding *why* something is true, not just *that* it is.

> Example 1: "Bumrah's numbers in SENA countries (avg 22.4, SR 46.2 over 18 tests) are being ignored in this conversation. The conditions at Lord's heavily favor swing, and he's the only Indian pacer who's been consistently effective there. Dropping him would be the wrong call."

> Example 2: "People are forgetting that Kohli's conversion rate (50s to 100s) has dropped from 47% pre-2020 to 31% since. That's not a slump — that's a structural change in how he's batting in the 70–90 range. The feet aren't moving the same way they did against short-pitched bowling."

**`hot_take`**
A bold, confident opinion stated without supporting evidence. The post asserts a claim — often provocative or contrarian — but doesn't show its work. There may be a stat or a name dropped, but it functions as decoration rather than as reasoning. The framing is usually confident, dismissive, or hyperbolic.

> Example 1: "Rohit is finished. He hasn't looked like a Test player in two years. India needs to move on before the Champions Trophy."

> Example 2: "England's Bazball is a fraud tactic that will never work in the subcontinent. Anyone who thinks otherwise doesn't understand real cricket."

**`reaction`**
An immediate emotional or social response to something that just happened — a wicket, a six, a selection announcement, a controversy. The post is expressive rather than argumentative. There's no claim being made beyond the feeling itself. These posts are valid and often high-engagement, but they're not taking a position — they're registering a moment.

> Example 1: "WHAT A CATCH. Absolute screamer. I'm losing my mind."

> Example 2: "Can't believe they dropped Jadeja for this. Heartbroken."

---

## 3. Hard Edge Cases

### Anticipated Hard Case: Stat-Backed Hot Take

**The post:** "Shubman Gill averages 28 in overseas Tests. He's not a Test-quality opener. The selectors are delusional."

**Why it's ambiguous:** It cites a real stat (averages 28 overseas), which looks like `analysis`. But the stat is cherry-picked for effect, the conclusion ("delusional") is asserted not argued, and no context is provided (sample size, conditions, age, comparable players at the same stage).

**Decision rule:** If removing the statistic would leave a claim that still feels confident and complete — i.e., the stat is supporting the *feeling* rather than driving the *reasoning* — label it `hot_take`. A genuine `analysis` post would engage with the stat: compare it to a benchmark, note its limitations, or trace what it implies structurally. Decoration ≠ evidence.

### Anticipated Hard Case: Emotional Post With a Buried Argument

**The post:** "I'm so tired of people defending this team. They've lost 7 of their last 10 away Tests. Nothing is working."

**Why it's ambiguous:** There's frustration (feels like `reaction`) but also a factual claim being used argumentatively. 

**Decision rule:** If the emotional language is the frame but there's an actual claim being advanced (however loosely), label it `hot_take`. `reaction` is reserved for posts that are *purely* expressive — responding to a moment with feeling, not making any argument at all.

---

## 4. Data Collection Plan

**Source:** Reddit, via PRAW (Python Reddit API Wrapper)  
**Subreddits:** r/Cricket and r/IndianCricket  
**Target:** 220+ labeled examples (buffer above 200 minimum)

**Per-label targets:**
| Label | Target count | % of dataset |
|-------|-------------|--------------|
| analysis | 70 | ~32% |
| hot_take | 80 | ~36% |
| reaction | 70 | ~32% |

**Collection strategy:**
- Pull top-level comments from recent match threads, selection discussion threads, and controversy threads
- Filter: 20–400 characters (eliminates one-word replies and walls of text that aren't representative)
- Deduplicate by text; strip usernames and links before labeling
- Collect ~300 raw examples, annotate, then trim to balance

**If a label is underrepresented after 150 examples:**
- For `analysis`: specifically target post-match discussion threads and "deep dive" flairs
- For `reaction`: target live match threads (these are overwhelmingly reactive)
- For `hot_take`: target selection announcement threads and controversy discussions

---

## 5. Evaluation Metrics

**Primary metrics:** Per-class F1 score + overall accuracy

**Why F1 per class and not just accuracy:**  
The dataset will be roughly balanced (~33% per class), but a model that learns to predict `hot_take` most of the time could still hit 36% accuracy. F1 per class exposes this — a class with F1 near zero is a label the model hasn't learned, regardless of overall accuracy.

**Secondary:** Confusion matrix  
The most useful diagnostic. I expect the model to confuse `hot_take` ↔ `analysis` (both are opinion posts, just with different evidence density) and to perform well on `reaction` (stylistically distinct). The confusion matrix will confirm or contradict this.

**Why not precision/recall alone:**  
In a three-way classification task with no strong asymmetry between false positives and false negatives, F1 is the right summary statistic. If I were building a moderation tool where one error type was much worse than the other, I'd weight precision or recall separately — but for this evaluation task, F1 is appropriate.

---

## 6. Definition of Success

**Minimum acceptable:** Fine-tuned model achieves ≥ 65% overall accuracy on the test set AND no per-class F1 below 0.50.

**"Good enough for deployment":** Overall accuracy ≥ 72%, all per-class F1 ≥ 0.60, fine-tuned model outperforms zero-shot baseline by ≥ 10 percentage points overall.

**Why these thresholds:**  
This is a subjective three-class task on informal internet text. 72% accuracy on a three-class problem means the model is right nearly 3x more often than random chance and is meaningfully better than a strong LLM with no training signal. A per-class F1 floor of 0.60 ensures the model has learned all three distinctions, not just the easiest one.

---

## 7. AI Tool Plan

**Label stress-testing (before annotation):**  
Feed label definitions to Claude with this prompt: *"Generate 10 cricket Reddit comments that sit at the boundary between `analysis` and `hot_take`. Each should be plausibly labelable as either. Don't label them — just generate them."* Use the output to pressure-test my decision rules before annotating 200 examples.

**Annotation assistance (during data collection):**  
Use Claude to pre-label batches of 20 examples at a time, providing it the full label definitions and decision rules. Review every pre-labeled example individually and correct any that I disagree with. Track the correction rate — if I'm overriding more than 30% of pre-labels, the definitions need tightening.

**Failure pattern analysis (after evaluation):**  
After fine-tuning, paste all misclassified test examples into Claude with the prompt: *"Group these by any pattern you notice — post length, use of statistics, sarcasm, specific topic (selection vs. match performance), etc. Don't just list them — identify generalizable themes."* Verify the patterns against the raw examples myself before writing them into the evaluation report.

---

## Stretch Features

*(To be updated before starting any stretch feature)*

- [ ] Error Pattern Analysis
- [ ] Deployed Interface
- [ ] Inter-Annotator Reliability
- [ ] Confidence Calibration
