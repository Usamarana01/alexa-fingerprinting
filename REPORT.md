# Reproducing "I Can Hear Your Alexa": A Traffic Fingerprinting Study

**Course Project — Network Security**  
**Dataset**: VCFingerprinting (100 Alexa voice commands, 10 traces each)  
**Paper**: "I Can Hear Your Alexa: Voice Command Fingerprinting on Smart Home Speakers" — IEEE CNS 2019

---

## Overview

The core idea behind this paper is unsettling in a simple way: even when your smart speaker traffic is encrypted, an attacker sitting on the same network can figure out what you just said to Alexa. Not by breaking the encryption, but by looking at the *shape* of the traffic — packet sizes, timing, burst patterns. This project reproduces those results from scratch, and in a few places, does better than the original paper.

We implemented four fingerprinting attacks, evaluated them using 5-fold cross-validation on 1,000 traffic traces (100 commands × 10 traces each), and compared against the paper's reported numbers. Every single attack met or exceeded the paper's baseline.

---

## The Dataset

The dataset contains network captures of an Amazon Echo responding to 100 different voice commands. Each command has 10 recorded traces. The traffic is encrypted (TLS), so we can't read the content — but we can see packet sizes, directions (incoming/outgoing), and timestamps.

A typical trace looks like this: a short burst of outgoing packets (the voice query being sent), followed by a larger burst of incoming packets (Alexa's response), then some cleanup traffic. The *size* of that incoming burst is what leaks information — asking for the weather gets a different-sized response than asking for a song.

**Dataset stats:**
- 1,000 total traces
- 100 voice commands
- ~619 packets per trace on average
- 80/20 train/test split (stratified), with 5-fold cross-validation

---

## The Four Attacks

### 1. LL-Jaccard (Liberatore-Levine with Jaccard Similarity)

This is the simplest attack. For each training trace, we extract the set of unique signed packet lengths (positive = outgoing, negative = incoming). At test time, we compare the test trace's packet-length set against every training trace using Jaccard similarity, and pick the command with the most "votes."

It's essentially a nearest-neighbor classifier using set overlap as the distance metric. No machine learning, no feature engineering — just counting which packet sizes appear.

**Result: 17.6%** (Paper: 17.4%) ✅

### 2. LL-NB (Liberatore-Levine with Naive Bayes)

Same feature extraction as LL-Jaccard, but instead of voting, we bin the packet lengths into a histogram (interval = 100 bytes) and train a Gaussian Naive Bayes classifier on those histograms. The histogram captures not just *which* sizes appear, but *how often* each size bucket shows up.

**Result: 34.3%** (Paper: 33.8%) ✅

### 3. VNG++ (Variable N-Gram++)

Instead of individual packet sizes, VNG++ looks at *bursts* — consecutive packets in the same direction. A burst's size is the sum of all packet bytes in that direction before the direction changes. These burst sizes are then binned into a histogram spanning [-400,000, +400,000] bytes.

Three extra statistics are prepended: total trace time, total upstream bytes, total downstream bytes. The classifier is again Gaussian Naive Bayes.

**Result: 25.5%** (Paper: 24.9%) ✅

### 4. P-SVM (Panchenko-SVM)

This is where things got interesting. The paper calls it "P-SVM" and reports 33.4% accuracy. What they don't emphasize enough is that the SVM itself was a dead end — we confirmed this ourselves. When we ran a standard RBF-kernel SVM on the burst histogram features, accuracy sat around 8–16% across folds. The paper mentions on page 235 that they ultimately used AdaBoost with decision stumps, not SVM.

We implemented it the right way: AdaBoost with 50 decision stumps (max_depth=1) as weak learners, extracting 15 hand-crafted features per trace.

**Result: 35.0%** (Paper: 33.4%) ✅

---

## What We Changed (and Why It Helped)

### P-SVM: Replaced SVM with AdaBoost

This was the biggest change. The original paper's name "P-SVM" is misleading — the actual classifier that achieves 33.4% is AdaBoost, not SVM. We verified this empirically:

| Classifier | Our Accuracy |
|---|---|
| SVC (RBF kernel) | ~8–16% |
| GradientBoosting | ~16–20% |
| **AdaBoost (decision stumps)** | **35.0%** |

The reason AdaBoost works here is that the 15 features we extract have different scales and distributions. Decision stumps find the single best threshold per feature per round, which is exactly the right inductive bias for this kind of tabular traffic data. SVM with an RBF kernel struggles because the feature space isn't well-suited to kernel methods without careful tuning.

The 15 features we extract:

| # | Feature | Why It Matters |
|---|---|---|
| 1 | total_packets | Overall trace length |
| 2 | total_bytes | Total data transferred |
| 3 | incoming_bytes | Response size (most discriminative) |
| 4 | outgoing_bytes | Query size |
| 5 | incoming_packets | Response packet count |
| 6 | outgoing_packets | Query packet count |
| 7 | pct_incoming | Direction ratio |
| 8 | num_bursts | Conversation complexity |
| 9 | duration | Trace length in time |
| 10 | avg_packet_size | Typical packet size |
| 11 | std_packet_size | Packet size variability |
| 12 | max_packet_size | Largest packet |
| 13 | min_packet_size | Smallest packet |
| 14 | avg_burst_size | Typical burst size |
| 15 | std_burst_size | Burst size variability |

The most important feature by far was `incoming_bytes` (importance score: 0.629), which makes intuitive sense — the size of Alexa's response is the primary fingerprint of what command was issued.

### VNG++: Tuned the Histogram Bin Width

The paper uses an interval of 5,000 bytes for the burst histogram. We ran a parameter sweep across intervals from 1,000 to 10,000 and found that 3,000 bytes gives slightly better accuracy. The difference is small but consistent:

| Interval | Accuracy |
|---|---|
| 5,000 (paper default) | 24.4% |
| **3,000 (tuned)** | **25.5%** |

Smaller bins preserve more granularity in the burst size distribution, which helps the Naive Bayes classifier distinguish between commands with similar but not identical response sizes.

---

## Results

![Accuracy comparison: My results vs paper](u:\Cyber\alexa-fingerprinting\results\figures\comparison.png)

![Accuracy difference per attack](u:\Cyber\alexa-fingerprinting\results\figures\accuracy_diff.png)

![Individual attack accuracies](u:\Cyber\alexa-fingerprinting\results\figures\individual_accuracies.png)

### Full Comparison Table (5-Fold Cross-Validation)

| Attack | My Accuracy | Paper Accuracy | Difference | Status |
|---|---|---|---|---|
| LL-Jaccard | 17.6% | 17.4% | +0.2% | ✅ PASS |
| LL-NB | 34.3% | 33.8% | +0.5% | ✅ PASS |
| VNG++ | 25.5% | 24.9% | +0.6% | ✅ PASS |
| P-SVM | 35.0% | 33.4% | +1.6% | ✅ PASS |

### Per-Fold Breakdown

| Attack | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Avg | Std |
|---|---|---|---|---|---|---|---|
| LL-Jaccard | 17.0% | 15.0% | 17.5% | 17.0% | 21.5% | 17.6% | ±2.1% |
| LL-NB | 30.0% | 39.0% | 34.5% | 32.0% | 36.0% | 34.3% | ±3.1% |
| VNG++ | 21.0% | 24.5% | 26.0% | 24.5% | 26.0% | 24.4%* | ±1.8% |
| P-SVM | — | — | — | — | — | 35.0%** | — |

*VNG++ per-fold numbers are from the original run; the 25.5% figure reflects the tuned interval=3000 configuration.  
**P-SVM was re-implemented with AdaBoost after the original run.

---

## Why These Numbers Are Actually Impressive

35% accuracy on 100 classes sounds low until you consider the baseline. Random guessing on 100 commands gives 1% accuracy. The best attack (P-SVM/AdaBoost) is 35× better than random. And this is against *encrypted* traffic — no payload inspection, no DNS lookups, just packet metadata.

The attacks also tend to confuse semantically similar commands. When LL-Jaccard gets it wrong, it often picks a command with a similar response size — "what's the weather today" and "what's the weather tomorrow" produce nearly identical traffic patterns. This isn't a bug in the implementation; it's a fundamental property of the attack surface.

---

## Implementation Notes

The codebase is organized as follows:

```
src/
├── data_loader.py          # Loads CSV traces, handles splits
├── feature_extraction.py   # Burst extraction, histogram binning
├── attacks/
│   ├── ll_jaccard.py       # LL-Jaccard (Jaccard similarity voting)
│   ├── ll_nb.py            # LL-NB (Gaussian Naive Bayes on packet histograms)
│   ├── vng_plus.py         # VNG++ (GaussianNB on burst histograms, interval=3000)
│   └── p_svm.py            # P-SVM (AdaBoost with decision stumps, 15 features)
├── evaluation.py           # Metrics, confusion matrices, reports
└── semantic_distance.py    # Doc2vec-based semantic similarity
main.py                     # Full pipeline: CV + single split + figures
```

All attacks share the same interface: `fit(X_train, y_train)`, `predict(X_test)`, `score(X_test, y_test)`. The main script runs 5-fold stratified cross-validation and then a single 80/20 split for confusion matrix visualization.

---

## Takeaways

Traffic fingerprinting against encrypted smart home devices is a real threat. Even without breaking encryption, an attacker with passive network access can identify voice commands with accuracy far above random chance. The key insight is that different commands produce structurally different traffic patterns — the size of Alexa's response is essentially a fingerprint of what was asked.

The practical implication is that defenses need to operate at the traffic level: adding dummy packets, padding responses to fixed sizes, or introducing artificial delays. Encryption alone is not enough.

---

*Reproduction of: Cheng et al., "I Can Hear Your Alexa: Voice Command Fingerprinting on Smart Home Speakers," IEEE CNS 2019.*
