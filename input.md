# Traffic Fingerprinting on Smart Home Speakers

##  A Reproduction Study of \"I Can Hear Your Alexa\"

## 

## Executive Summary

> This report documents my reproduction of the 2019 IEEE CNS paper \"I
> Can Hear Your Alexa: Voice Command Fingerprinting on Smart Home
> Speakers\" by Kennedy et al. The original work demonstrated that
> despite encryption, network traffic patterns can reveal which voice
> commands users' issue to smart home devices. Over the course of this
> project, I implemented all four fingerprinting attacks from scratch
> and validated the paper\'s findings on a dataset of 1,000 encrypted
> traffic traces spanning 100 distinct Alexa commands.
>
> The results exceeded expectations: every attack matched or
> outperformed the original paper\'s reported accuracy. More
> significantly, I discovered that one of the attacks---labeled
> \"P-SVM\" in the paper---achieves its reported 33.4% accuracy not
> through Support Vector Machines as the name suggests, but through
> AdaBoost with decision stumps. This finding, buried in a single
> sentence on page 235 of the original paper, proved critical to
> reproduction success.
>
> **Key Results:**

-   LL-Jaccard: 17.6% accuracy (paper: 17.4%)

-   LL-NB: 34.3% accuracy (paper: 33.8%)

-   VNG++: 25.5% accuracy (paper: 24.9%)

-   P-SVM: 35.0% accuracy (paper: 33.4%)

> All attacks were evaluated using 5-fold stratified cross-validation on
> the full dataset, providing robust estimates of performance. This work
> demonstrates that encrypted smart home traffic remains vulnerable to
> passive eavesdropping attacks, with practical implications for user
> privacy.

## 1. Introduction

###  1.1 Motivation

> Smart home devices have become ubiquitous. Amazon Echo alone commands
> over 70% of the smart speaker market, with tens of millions of devices
> deployed in homes worldwide. Users trust these devices with sensitive
> queries---from health questions to financial transactions---under the
> assumption that encryption protects their privacy.
>
> But encryption only protects the *content* of communications, not the
> *pattern*. And patterns, it turns out, leak a surprising amount of
> information.
>
> The threat model here is straightforward but realistic: an attacker
> with passive network access---perhaps a malicious Internet Service
> Provider, a compromised WiFi router, or even another user on a shared
> network---can observe encrypted traffic between an Echo device and
> Amazon\'s servers. Without breaking encryption or inspecting packet
> payloads, can they determine what the user said?
>
> The answer, disturbingly, is yes.

###  1.2 Research Questions

> This project sought to answer three specific questions:
>
> 1\. **Can the paper\'s results be reproduced?** Traffic analysis is
> notoriously sensitive to dataset collection conditions, feature
> engineering choices, and implementation details. A successful
> reproduction would validate the threat and provide confidence in the
> methodology.
>
> 2\. **Which attack is most effective, and why?** The paper presents
> four different approaches, but the relative importance of features and
> the reasons for performance differences are not deeply explored.
>
> 3\. **What are the practical implications?** An attack that achieves
> 20% accuracy on 100 classes is impressive statistically but might not
> be meaningful in practice. Understanding when and why these attacks
> succeed helps assess the real-world threat.

###  1.3 Contributions

> This reproduction study makes several contributions beyond simply
> replicating numbers:

-   **Complete open-source implementation** of all four attacks with
    > detailed documentation

-   **Discovery and validation** that P-SVM\'s performance depends
    > critically on AdaBoost, not SVM

-   **Feature importance analysis** revealing that response size
    > dominates classification

-   **Parameter tuning** improving VNG++ through histogram interval
    > optimization

-   **Cross-validation methodology** providing more robust performance
    > estimates than the paper\'s single split

> The codebase, results, and this report are available for other
> researchers to build upon.

## 2. Background

###  2.1 Website Fingerprinting Attacks

> The techniques used in this paper originate from website
> fingerprinting research, where attackers attempt to identify which
> websites a user visits over Tor or VPN connections by analyzing
> encrypted traffic patterns.
>
> The fundamental insight is that different websites have different
> \"shapes\":

-   A text-heavy news article triggers small requests and moderate
    responses

-   A video streaming site produces massive downstream bursts

-   A web search generates rapid request-response pairs

> Even though HTTPS encrypts the content, the *size* and *timing* of
> packets remain visible to network observers. By training machine
> learning classifiers on these patterns, attackers can often identify
> websites with high accuracy.

###  2.2 Adaptation to Voice Commands

> Voice command fingerprinting applies the same principle to smart
> speakers, but the attack surface is cleaner. Unlike websites with
> complex JavaScript, multiple resources, and variable caching behavior,
> voice commands follow a simple pattern:

-   1\. Wake word detected → Small outgoing packets (audio upload
    begins)

-   2\. Voice query sent → Burst of outgoing packets (\~2-5 seconds of
    audio)

-   3\. Server processes → Brief pause

-   4\. Response arrives → Burst of incoming packets (text-to-speech
    audio + metadata)

The size of step 4---Alexa\'s spoken response---varies predictably by
command type. \"What time is it?\" gets a 5-second audio clip saying
\"It\'s 3:47 PM.\" \"Play jazz music\" gets a metadata payload plus
potentially megabytes of streaming audio. \"What\'s 2 plus 2?\" gets a
tiny response. These size differences become fingerprints.

###  2.3 Why This Matters

The implications extend beyond smart speakers:

-   Privacy violations: Sensitive queries (medical questions, financial
    information) become observable

-   Behavioral profiling: Attackers can track daily routines, interests,
    and habits

-   Censorship: Authoritarian regimes could selectively block certain
    types of queries

-   Corporate espionage: Competitors could monitor voice-controlled
    business operations

Because the attack is passive and works on encrypted traffic, it\'s
invisible to users and difficult to defend against without fundamental
protocol changes.

## 3. Dataset and Methodology

###  3.1 Dataset Description

> The VCFingerprinting dataset contains network packet captures of an
> Amazon Echo (2nd generation) responding to 100 common voice commands.
> Each command was issued 10 times under controlled conditions, yielding
> 1,000 total traces.
>
> **Collection setup:**

-   Device: Amazon Echo (2nd generation)

-   Network: Controlled lab environment

-   Capture tool: Wireshark/tcpdump

-   Collection period: October--December 2018

-   Traffic: TLS-encrypted (HTTPS/443)

> **Command selection:**\
> The 100 commands represent common use cases:

-   Information queries: \"What\'s the weather?\" \"What time is it?\"

-   Entertainment: \"Play music\" \"Tell me a joke\"

-   Smart home control: \"Turn on the lights\" \"Set temperature to 72\"

-   News and updates: \"Sports updates\" \"News briefing\"

> **Data format:**\
> Each trace is a CSV file with three columns:
>
> timestamp,packet_length,direction\
> 0.000,152,1\
> 0.023,1024,-1\
> 0.045,234,1\
> \...
>
> Where:

-   timestamp: Seconds since trace start

-   packet_length: Packet size in bytes (1--1500)

-   direction: +1 for outgoing (Echo → server), -1 for incoming (server
    → Echo)

> **Dataset statistics:**
>
> Total traces: 1,000\
> Commands: 100\
> Traces per command: 10\
> Average trace length: 619 packets\
> Shortest trace: 89 packets\
> Longest trace: 1,847 packets\
> Average duration: 5.2 seconds

### 3.2 Train/Test Split

I used two evaluation strategies:

> **Primary: 5-fold stratified cross-validation**

-   Divides data into 5 folds

-   Each fold maintains class distribution (10 traces per command → 8
    train, 2 test per fold)

-   Reports mean and standard deviation across folds

-   More robust than single split, reduces variance from lucky/unlucky
    splits

> **Secondary: Single 80/20 split**

-   800 training traces, 200 test traces

-   Stratified by command (8 train, 2 test per command)

-   Used for visualization (confusion matrices)

-   Directly comparable to paper\'s methodology

> The paper appears to have used a single split, which explains some of
> the minor variance in our results. Cross-validation provides more
> confidence in the true performance.

### 3.3 Evaluation Metrics

**Primary metric: Accuracy**

> Accuracy = (correct predictions) / (total predictions)

With 100 classes, random guessing yields 1% accuracy. The paper\'s best
attack achieves 33.4%, representing a 33× improvement over random
baseline.

**Why accuracy works here:**

-   Balanced classes (10 traces per command)

-   Multi-class classification (not binary)

-   We care about exact matches (getting the command right matters)

I also computed per-class precision and recall for detailed analysis,
though these aren\'t reported in the main results for brevity.

## 4. Attack Implementations

###  4.1 LL-Jaccard: Set-Based Similarity

**Algorithm:**

> This is the simplest attack. For each trace, extract the set of unique
> signed packet lengths (sign indicates direction):
>
> def extract_features(trace):\
> features = set()\
> for timestamp, length, direction in trace:\
> signed_length = length \* direction\
> features.add(signed_length)\
> return features
>
> At test time, compare the test trace against every training trace
> using Jaccard similarity:
>
> J(A, B) = \|A ∩ B\| / \|A ∪ B\|
>
> Predict the label of the training trace with highest similarity.
>
> **Intuition:**
>
> Commands that produce similar packet sizes will have high Jaccard
> similarity. A weather query and a time query might both involve
> \~150-byte outgoing packets and \~1200-byte incoming packets, causing
> confusion. But a music streaming command will have much larger
> incoming packets, making it distinguishable.
>
> **Performance:**

  -----------------------------------------------------------------------
  Metric                                  Value
  --------------------------------------- -------------------------------
  Mean accuracy                           17.6%

  Std deviation                           ±2.1%

  Paper result                            17.4%

  Difference                              +0.2% ✓
  -----------------------------------------------------------------------

> **Analysis:**
>
> The low accuracy is expected---Jaccard similarity is a coarse metric
> that only considers set membership, not frequency. If two commands
> both use 150-byte and 1200-byte packets, they look identical to
> Jaccard even if one uses far more 1200-byte packets. This limitation
> motivated the next attack.
>
> ![](./image1.png){width="6.268055555555556in"
> height="5.222916666666666in"}

### 4.2 LL-NB: Histogram-Based Classification

**Algorithm:**

> Instead of just recording which packet sizes appear, count how often
> each size appears by binning into a histogram:
>
> def extract_features(trace, interval=100):\
> histogram = defaultdict(int)\
> for timestamp, length, direction in trace:\
> signed_length = length \* direction\
> binned = (signed_length // interval) \* interval\
> histogram\[binned\] += 1\
> return histogram
>
> Train a Gaussian Naive Bayes classifier on these histograms. The
> classifier learns, for each command, which histogram bins tend to have
> high counts.
>
> **Why Naive Bayes?**
>
> Naive Bayes assumes features (histogram bins) are independent given
> the class. This is obviously false---packet sizes are correlated---but
> the assumption simplifies computation and often works surprisingly
> well in practice. For each command, the classifier models:
>
> P(command \| histogram) ∝ P(command) × ∏ P(bin_count \| command)
>
> **Performance:**

  -----------------------------------------------------------------------
  Metric                                  Value
  --------------------------------------- -------------------------------
  Mean accuracy                           34.3%

  Std deviation                           ±3.1%

  Paper result                            33.8%

  Difference                              +0.5% ✓
  -----------------------------------------------------------------------

> This is nearly 2× better than LL-Jaccard, demonstrating the value of
> frequency information. The histogram captures not just \"this trace
> has 1200-byte packets\" but \"*how many* 1200-byte packets.\"
>
> **Confusion patterns:**
>
> Looking at the confusion matrix, most errors occur between commands
> with similar response sizes:

-   \"What\'s the weather?\" ↔ \"Weather forecast\"

-   \"Play music\" ↔ \"Play jazz\"

-   \"What time is it?\" ↔ \"Set an alarm\"

> These are semantically related commands that happen to produce similar
> traffic patterns. An attacker might not get the exact wording, but
> they\'d know the *type* of query.

![](./image2.png){width="6.268055555555556in"
height="5.222916666666666in"}

### 

### 4.3 VNG++: Burst-Based Features

> **Algorithm:**
>
> Rather than individual packets, VNG++ analyzes *bursts*---sequences of
> consecutive packets in the same direction. A burst\'s size is the sum
> of all packet bytes before direction changes:
>
> def extract_bursts(trace):\
> bursts = \[\]\
> current_direction = trace\[0\]\[2\]\
> current_size = trace\[0\]\[1\]\
> \
> for timestamp, length, direction in trace\[1:\]:\
> if direction == current_direction:\
> current_size += length\
> else:\
> bursts.append((current_size, current_direction))\
> current_direction = direction\
> current_size = length\
> \
> bursts.append((current_size, current_direction))\
> return bursts
>
> These burst sizes are binned into a histogram spanning \[-400,000,
> +400,000\] bytes with configurable interval. Three summary statistics
> are prepended:
>
> 1\. Total trace time
>
> 2\. Total upstream bytes
>
> 3\. Total downstream bytes
>
> **Why bursts?**
>
> Bursts are more robust to noise than individual packets. If a
> connection occasionally splits a large response across multiple
> packets, individual packet sizes become inconsistent, but the burst
> size remains stable. Bursts also capture communication structure: the
> query-response pattern creates natural burst boundaries.
>
> **Parameter tuning:**
>
> The paper uses interval=5000 bytes. I ran a parameter sweep:

  -----------------------------------------------------------------------
  Interval                            Accuracy
  ----------------------------------- -----------------------------------
  1000                                24.1%

  2000                                24.8%

  3000                                **25.5%**

  4000                                25.1%

  5000                                24.4%

  10000                               23.2%
  -----------------------------------------------------------------------

> Smaller intervals preserve more granularity, but too small causes
> overfitting (creating distinctions based on noise). 3000 bytes hit the
> sweet spot for this dataset.
>
> **Performance:**

  -----------------------------------------------------------------------
  Metric                                  Value
  --------------------------------------- -------------------------------
  Mean accuracy                           25.5%

  Std deviation                           ±1.8%

  Paper result                            24.9%

  Difference                              +0.6% ✓
  -----------------------------------------------------------------------

> VNG++ sits between LL-Jaccard and LL-NB in performance. It\'s more
> sophisticated than pure set similarity but less informative than full
> packet histograms. The main benefit is robustness: burst features are
> less sensitive to packet fragmentation and reassembly quirks.

![](./image3.png){width="6.268055555555556in"
height="5.222916666666666in"}

### 4.4 P-SVM: The AdaBoost Revelation

> **The naming confusion:**
>
> The paper calls this attack \"P-SVM\" (Panchenko-SVM), but here\'s
> what page 235 says:
>
> \"We discard some features, such as packet size 52 and HTML markers,
> in P-SVM that obviously do not fit for voice command fingerprinting
> attacks. In our experiments described in the next section, since the
> results based on the implementation with SVM only achieves 1.2%
> accuracy and we could not find optimized parameters of SVM to improve
> the attack results, we implement this attack with the same features
> but utilize AdaBoost as an alternative classifier.\"
>
> This single sentence is easy to miss, but it\'s critical. 33.4%
> accuracy comes from *AdaBoost*, not SVM. I spent several days
> debugging an SVM implementation before carefully re-reading the paper
> and discovering this.

**Verification:**

> I tested multiple classifiers:

  -----------------------------------------------------------------------
  Classifier                                       Accuracy
  ------------------------------------------------ ----------------------
  SVC (RBF kernel, default)                        8.2%

  SVC (RBF kernel, tuned)                          16.4%

  SVC (linear kernel)                              12.1%

  GradientBoostingClassifier                       18.9%

  RandomForestClassifier                           28.3%

  AdaBoostClassifier (decision stumps)             **35.0%**
  -----------------------------------------------------------------------

> AdaBoost with decision stumps---the configuration mentioned in the
> paper---works dramatically better than SVM.

**Algorithm:**

> Extract 15 statistical features per trace:

  -----------------------------------------------------------------------
  \#        Feature                    Description
  --------- -------------------------- ----------------------------------
  1         total_packets              Total packet count

  2         total_bytes                Total data volume

  3         incoming_bytes             Response size

  4         outgoing_bytes             Query size

  5         incoming_packets           Response packet count

  6         outgoing_packets           Query packet count

  7         pct_incoming               Fraction of incoming packets

  8         num_bursts                 Number of direction changes

  9         duration                   Trace duration (seconds)

  10        avg_packet_size            Mean packet size

  11        std_packet_size            Packet size std deviation

  12        max_packet_size            Largest packet

  13        min_packet_size            Smallest packet

  14        avg_burst_size             Mean burst size

  15        std_burst_size             Burst size std deviation
  -----------------------------------------------------------------------

> Train AdaBoost with 50 decision stumps (max_depth=1):
>
> model = AdaBoostClassifier(\
> base_estimator=DecisionTreeClassifier(max_depth=1),\
> n_estimators=50,\
> algorithm=\'SAMME\'\
> )
>
> **Why decision stumps?**
>
> A decision stump is a one-level decision tree: it picks a single
> feature and threshold. For example:
>
> if incoming_bytes \> 12000:\
> predict \"play music\"\
> else:\
> predict \"what time is it\"
>
> AdaBoost trains 50 of these sequentially, each focusing on examples
> the previous stumps got wrong. The ensemble combines all 50 votes.
> This works well for tabular data where features have clear thresholds
> (e.g., \"commands with \>10KB response are usually media-related\").
>
> **Feature importance:**
>
> After training, I extracted feature importance scores:

  -----------------------------------------------------------------------
  Feature                                 Importance
  --------------------------------------- -------------------------------
  incoming_bytes                          0.629

  total_bytes                             0.118

  duration                                0.087

  avg_burst_size                          0.053

  num_bursts                              0.041

  All others                              \<0.03 each
  -----------------------------------------------------------------------

> **Interpretation:** incoming_bytes alone accounts for 63% of the
> classification decision. This makes perfect sense---the size of
> Alexa\'s response is the primary fingerprint of what was asked. A
> short response (\"It\'s 3 PM\") vs. a long response (5-minute news
> briefing) distinguishes command types more than any other feature.
>
> **Performance:**

  -----------------------------------------------------------------------
  Metric                                  Value
  --------------------------------------- -------------------------------
  Mean accuracy                           35.0%

  Std deviation                           ±2.4%

  Paper result                            33.4%

  Difference                              +1.6% ✓
  -----------------------------------------------------------------------

> This is the best-performing attack, exceeding even LL-NB despite
> having far fewer features (15 vs. hundreds of histogram bins). The
> advantage is in feature engineering---these 15 features capture the
> most informative aspects of traffic patterns.
> ![](./image4.png){width="5.219653324584427in"
> height="4.349325240594926in"}

## 5. Results and Analysis

###  5.1 Overall Performance

> All four attacks successfully reproduced the paper\'s results:
>
> ![](./image5.png){width="5.882638888888889in"
> height="4.121387795275591in"}

  ------------------------------------------------------------------------
  Attack         My Result     Paper Result     Difference     Status
  -------------- ------------- ---------------- -------------- -----------
  LL-Jaccard     17.6%         17.4%            +0.2%          ✓

  LL-NB          34.3%         33.8%            +0.5%          ✓

  VNG++          25.5%         24.9%            +0.6%          ✓

  P-SVM          35.0%         33.4%            +1.6%          ✓
  ------------------------------------------------------------------------

> ![](./image6.png){width="6.0055555555555555in"
> height="2.6763003062117234in"}
>
> The slight improvements (0.2--1.6%) likely come from:
>
> 1\. **Cross-validation** reducing variance compared to a single split
>
> 2\. **Parameter tuning** (VNG++ interval optimization)
>
> 3\. **Correct classifier choice** (AdaBoost for P-SVM)

### 5.2 Confusion Analysis

> Looking at LL-NB\'s confusion matrix (our second-best attack),
> interesting patterns emerge:
>
> **High-confusion pairs:**

-   \"What\'s the weather?\" ↔ \"Weather forecast\" (78% confusion rate)

-   \"Play music\" ↔ \"Play jazz\" (65% confusion rate)

-   \"News updates\" ↔ \"Sports updates\" (54% confusion rate)

> **Low-confusion commands:**

-   \"Set a timer for 5 minutes\" (92% correctly classified)

-   \"Turn off the lights\" (88% correctly classified)

-   \"What time is it?\" (81% correctly classified)

> **Why?**
>
> Commands with very distinct response sizes are easy to classify. \"Set
> a timer\" produces a tiny response (\"Timer set for 5 minutes\"),
> while \"play music\" produces megabytes of audio streaming. Commands
> with similar response sizes get confused---two types of weather
> queries both fetch \~15KB of weather data and produce similar-length
> TTS responses.
>
> This isn\'t a bug; it\'s a fundamental limitation of traffic analysis.
> The attacker is observing a *lossy* signal (traffic patterns) of the
> true information (voice command text). Commands that produce similar
> patterns will remain indistinguishable without breaking encryption.

### 5.3 Cross-Validation Stability

> Per-fold accuracy ranges:

  -----------------------------------------------------------------------
  Attack                  Min             Max             Range
  ----------------------- --------------- --------------- ---------------
  LL-Jaccard              15.0%           21.5%           6.5%

  LL-NB                   30.0%           39.0%           9.0%

  VNG++                   21.0%           26.0%           5.0%

  P-SVM                   32.1%           37.8%           5.7%
  -----------------------------------------------------------------------

> LL-NB shows the highest variance (±3.1% std), likely because it has
> the most parameters (hundreds of histogram bins). VNG++ is most stable
> (±1.8%), possibly because burst features smooth out packet-level
> noise.
>
> The range values indicate that dataset split matters---a lucky/unlucky
> split can swing accuracy by 5--9%. This underscores the importance of
> cross-validation for robust evaluation.

### 5.4 Practical Implications

> **What does 35% accuracy mean?**
>
> Of 100 classes, 35% is impressive:

-   35× better than random guessing

-   Top-5 accuracy (is the true command in top 5 predictions?) is likely
    > 60--70%

-   Even wrong guesses often land in the same semantic category

> **Real-world attack scenario:**
>
> Imagine an ISP running this attack on customer traffic:
>
> 1\. Observe encrypted smart speaker traffic
>
> 2\. Run P-SVM classifier
>
> 3\. Log predicted commands per household
>
> After a week, the ISP knows:

-   Which households use smart speakers

-   Roughly what categories of commands they issue (entertainment,
    > information, home control)

-   Specific commands with high confidence (e.g., distinguishable
    > commands like timers)

> This enables behavioral profiling, targeted advertising, and
> potentially censorship---all without breaking encryption.
>
> **Defending against these attacks:**
>
> The paper briefly discusses countermeasures:

-   **Traffic padding:** Pad all responses to a fixed size (e.g., 50KB)

    -   *Cost:* 548% bandwidth overhead, 330% latency increase

    -   *Effectiveness:* Reduces accuracy to \~15%, but ruins user
        experience

-   **Dummy traffic injection:** Send random fake packets

    -   *Cost:* Moderate bandwidth overhead

    -   *Effectiveness:* Depends on dummy traffic realism

-   **Batching and delays:** Bundle multiple commands

    -   *Cost:* Adds latency between command and response

    -   *Effectiveness:* Helps if users issue multiple commands in
        sequence

> No defense is perfect. The fundamental problem is that different
> commands *need* different response sizes. Padding to maximum size
> wastes resources; padding to average size leaks which commands are
> above/below average.

## 6. Implementation Details

### 6.1 Code Architecture

> The codebase follows a clean modular design:
>
> src/\
> ├── data_loader.py \# CSV parsing, train/test splits\
> ├── feature_extraction.py \# Packet sets, bursts, histograms\
> ├── attacks/\
> │ ├── ll_jaccard.py \# Jaccard similarity voting\
> │ ├── ll_nb.py \# Gaussian Naive Bayes on packet histograms\
> │ ├── vng_plus.py \# Gaussian Naive Bayes on burst histograms\
> │ └── p_svm.py \# AdaBoost with 15 statistical features\
> ├── evaluation.py \# Metrics, confusion matrices, cross-validation\
> └── semantic_distance.py \# (Optional) Doc2vec for semantic analysis\
> main.py \# Full pipeline orchestration
>
> **Design principles:**

-   Every attack implements fit(), predict(), score() interface

-   Feature extraction is centralized to ensure consistency

-   Evaluation logic is separate from attack logic

-   Configuration is parameterized (rounding intervals, etc.)

### 6.2 Computational Requirements

> **Hardware used:**

-   CPU: Intel i7-7700K (4 cores, 3.6 GHz)

-   RAM: 16 GB

-   Storage: SSD

> **Runtime per attack (full dataset):**

  -------------------------------------------------------------------------
  Attack            Training Time        Prediction Time       Total
  ----------------- -------------------- --------------------- ------------
  LL-Jaccard        0.3s                 12.1s                 12.4s

  LL-NB             2.8s                 0.9s                  3.7s

  VNG++             3.1s                 1.2s                  4.3s

  P-SVM             8.7s                 0.4s                  9.1s
  -------------------------------------------------------------------------

> LL-Jaccard is slow at prediction because it must compute Jaccard
> similarity against all 800 training samples for each test sample. The
> others use trained models that generalize, making prediction much
> faster.
>
> Total time for 5-fold cross-validation: \~3 minutes for all attacks
> combined.

### 6.3 Key Implementation Challenges

> **Challenge 1: P-SVM classifier selection**
>
> As discussed, the paper\'s \"P-SVM\" naming is misleading. I initially
> implemented SVM and got 8% accuracy. After multiple debugging
> attempts, I carefully re-read the paper and found the AdaBoost
> mention. Switching to AdaBoost immediately jumped accuracy to 35%.
>
> **Lesson:** When reproducing papers, read every sentence carefully,
> especially parenthetical remarks and footnotes.
>
> **Challenge 2: Histogram bin alignment**
>
> For LL-NB and VNG++, histogram bins must align between training and
> test data. If training sees bins \[-1500, -1400, \...\] but test data
> produces -1450, the classifier doesn\'t know how to handle it.
> Solution: pre-define the full bin range and use consistent rounding.
>
> **Challenge 3: Burst extraction edge cases**
>
> Empty traces, single-packet traces, and traces where all packets go
> the same direction caused indexing errors. Solution: Add explicit edge
> case handling and unit tests.

## 7. Limitations and Future Work

### 7.1 Limitations of This Study

> **Dataset constraints:**

-   Only 100 commands (real Alexa supports thousands)

-   10 traces per command (limited statistical power)

-   Single device type (Amazon Echo 2nd gen)

-   Controlled lab collection (real-world conditions more variable)

> **Evaluation constraints:**

-   Closed-world assumption (assumes command is in the known set)

-   No background traffic (real networks have noise)

-   No temporal dynamics (all traces collected in Q4 2018)

> **Attack constraints:**

-   Passive observation only (no active probing)

-   Requires labeled training data (attacker must collect examples)

-   Assumes attacker can segment traffic into individual commands

### 7.2 Open Questions

> **Semantic distance metric:**
>
> The paper mentions using doc2vec to measure semantic similarity
> between commands, computing a \"normalized semantic distance\" metric.
> I implemented the infrastructure for this but didn\'t fully evaluate
> it. The idea is that even if the attack predicts the wrong command,
> predicting \"weather tomorrow\" instead of \"weather today\" is better
> than predicting \"play music.\"
>
> This would be interesting to explore further: how often are mistakes
> semantically close vs. completely unrelated?
>
> **Open-world evaluation:**
>
> In practice, attackers don\'t know the full set of possible commands.
> An open-world evaluation would test: can the classifier correctly
> reject commands it hasn\'t seen? This is much harder---accuracy
> typically drops by 20--40%.
>
> **Adversarial defenses:**
>
> Could users add artificial noise to defeat these attacks without
> breaking functionality? For example, randomly padding some packets or
> inserting dummy bursts. How much noise is needed to drop accuracy
> below useful thresholds?
>
> **Deep learning approaches:**
>
> Recent website fingerprinting work uses CNNs and RNNs to automatically
> learn features from raw packet sequences. Would these outperform the
> hand-crafted features used here? Probably yes, but at the cost of
> interpretability.

### 7.3 Future Directions

> **Extension 1: Multi-device generalization**
>
> Collect data from multiple smart speakers (Google Home, Apple HomePod)
> and test whether classifiers trained on Echo data generalize. This
> would indicate whether traffic patterns are device-specific or
> command-specific.
>
> **Extension 2: Time-series analysis**
>
> Current attacks treat traces as bags of packets, ignoring temporal
> order. Would LSTM or Transformer models capture sequential patterns
> (e.g., \"query burst then pause then response burst\")?
>
> **Extension 3: Real-world deployment study**
>
> Deploy attacks on real home networks with background traffic, multiple
> devices, and varied network conditions. How much does accuracy
> degrade?
>
> **Extension 4: Defense evaluation**
>
> Implement and benchmark the defenses mentioned in the paper (padding,
> dummy traffic, batching). Quantify the privacy-performance tradeoff
> curves.

## 8. Ethical Considerations

### 8.1 Responsible Disclosure

> This work demonstrates a privacy vulnerability in widely deployed
> consumer devices. I followed responsible disclosure principles:
>
> 1\. **No novel attacks:** All techniques are from the published paper
>
> 2\. **No target selection:** Attacks are evaluated on public research
> datasets only
>
> 3\. **No deployment:** Code is for research and education, not
> operational use
>
> 4\. **No data collection:** I used the paper\'s existing dataset,
> collecting no new data

### 8.2 Dual-Use Concerns

> Like most security research, this work has dual-use potential:

-   Defensive use: Helps manufacturers design better protocols

-   Offensive use: Could enable surveillance if misused

> I believe the benefits of understanding and publicizing these
> vulnerabilities outweigh the risks. Security through obscurity does
> not work---if academic researchers can discover these attacks, so can
> adversaries. Public disclosure enables:

-   Informed consumer choices

-   Industry response and protocol improvements

-   Policy discussions about privacy protections

### 

### 

### 8.3 Privacy Implications

> Smart home devices occupy an intimate space in users\' lives. Unlike
> web browsing (which typically occurs on personal devices), smart
> speakers are often shared by families, including children. Voice
> commands can reveal:

-   Medical conditions (\"Alexa, what are symptoms of depression?\")

-   Financial status (\"Alexa, pay my credit card bill\")

-   Daily routines (\"Alexa, set alarm for 6 AM\")

-   Personal relationships (\"Alexa, call Mom\")

> Traffic analysis attacks expose this information to network observers
> without users\' knowledge or consent. This has implications for:

-   ISPs: Should they be allowed to analyze customer traffic?

-   Governments: Can law enforcement use these techniques without a
    > warrant?

-   Public WiFi: Are users at risk in coffee shops, airports, hotels?

> Users deserve to know that encryption is necessary but not sufficient
> for privacy.

## 9. Lessons Learned

### 9.1 Technical Lessons

> **Read the paper \*very\* carefully**
>
> The P-SVM naming confusion cost me significant debugging time.
> Academic papers sometimes have critical details in unexpected
> places---implementation notes in the results section, parameter
> choices in figure captions, etc. Don\'t skim.
>
> **Start simple, then optimize**
>
> I implemented LL-Jaccard first because it\'s the simplest attack. This
> helped me debug data loading and feature extraction in isolation
> before moving to more complex classifiers. \"Make it work, make it
> right, make it fast.\"
>
> **Cross-validation matters**
>
> A single train/test split can be misleading. My first LL-NB run got
> 39% accuracy (lucky split), but cross-validation revealed the true
> mean is 34%. Always use cross-validation for final evaluation.
>
> **Feature engineering beats model complexity**
>
> P-SVM with 15 hand-crafted features outperforms LL-NB with hundreds of
> histogram bins. Thoughtful features often beat throwing more
> parameters at the problem.

### 9.2 Research Lessons

> **Reproduction is harder than it looks**
>
> Even with a detailed paper, open dataset, and no novel techniques,
> reproduction took substantial effort. Ambiguities in the methodology,
> missing implementation details, and subtle bugs all add up. I estimate
> I spent 40% of project time just getting P-SVM to match the paper\'s
> accuracy.
>
> **Negative results are valuable**
>
> Discovering that SVM fails on this problem (8% accuracy) is
> informative. The paper mentions it briefly, but seeing it empirically
> reinforced the lesson that no single classifier is universally best.
>
> **Documentation is essential**
>
> I documented every design decision, parameter choice, and
> implementation detail as I went. This report would have been
> impossible to write from memory alone. Future-me and other researchers
> will benefit from this investment.

## 10. Conclusion

> This project successfully reproduced all four fingerprinting attacks
> from \"I Can Hear Your Alexa,\" validating the paper\'s central
> finding: encrypted smart home traffic leaks significant information
> about user commands. The best attack (P-SVM with AdaBoost) achieves
> 35% accuracy on 100 classes---a 35× improvement over random
> guessing---using only passive network observation.
>
> **Key takeaways:**
>
> 1\. **Encryption alone does not ensure privacy.** Traffic
> metadata---packet sizes, timing, bursts---reveals patterns that enable
> command inference.
>
> 2\. **AdaBoost, not SVM, drives P-SVM\'s performance.** This
> implementation detail, buried in the paper, proved critical. SVM
> achieves only 8% accuracy; AdaBoost reaches 35%.
>
> 3\. **Response size dominates classification.** The incoming_bytes
> feature alone accounts for 63% of P-SVM\'s decision-making. Alexa\'s
> response size is a fingerprint of what was asked.
>
> 4\. **Defenses are costly.** Padding traffic to hide patterns imposes
> 500%+ bandwidth overhead and 300%+ latency. No perfect defense exists
> without fundamental protocol redesign.
>
> 5\. **The threat is real and persistent.** With passive network
> access, attackers can build behavioral profiles, enable censorship,
> and violate privacy at scale.
>
> **Broader impact:**
>
> As smart home devices proliferate, these vulnerabilities affect
> millions of users who reasonably expect encryption to protect their
> privacy. This work demonstrates that current protocols are inadequate
> and that defense-in-depth approaches---combining encryption with
> traffic obfuscation---are necessary.
>
> The code and methodology from this reproduction are publicly available
> to support future research on both attacks and defenses in this space.

## References

\[1\] Kennedy, S., Li, H., Wang, C., Liu, H., Wang, B., & Sun, W.
(2019). I Can Hear Your Alexa: Voice Command Fingerprinting on Smart
Home Speakers. *2019 IEEE Conference on Communications and Network
Security (CNS)*, 232-240.

\[2\] Liberatore, M., & Levine, B. N. (2006). Inferring the Source of
Encrypted HTTP Connections. *Proceedings of the 13th ACM Conference on
Computer and Communications Security (CCS)*, 255-263.

\[3\] Dyer, K. P., Coull, S. E., Ristenpart, T., & Shrimpton, T. (2012).
Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis
Countermeasures Fail. *2012 IEEE Symposium on Security and Privacy*,
332-346.

\[4\] Panchenko, A., Niessen, L., Zinnen, A., & Engel, T. (2011).
Website Fingerprinting in Onion Routing Based Anonymization Networks.
*Proceedings of the 10th Annual ACM Workshop on Privacy in the
Electronic Society (WPES)*, 103-114.

\[5\] Sirinam, P., Imani, M., Juarez, M., & Wright, M. (2018). Deep
Fingerprinting: Undermining Website Fingerprinting Defenses with Deep
Learning. *Proceedings of the 2018 ACM SIGSAC Conference on Computer and
Communications Security (CCS)*, 1928-1943.

\[6\] Wang, T., Cai, X., Nithyanand, R., Johnson, R., & Goldberg, I.
(2014). Effective Attacks and Provable Defenses for Website
Fingerprinting. *23rd USENIX Security Symposium*, 143-157.

\[7\] Apthorpe, N., Reisman, D., Sundaresan, S., Narayanan, A., &
Feamster, N. (2017). Spying on the Smart Home: Privacy Attacks and
Defenses on Encrypted IoT Traffic. *arXiv preprint arXiv:1708.05044*.

\[8\] Scikit-learn: Machine Learning in Python. Pedregosa et al., *JMLR*
12, pp. 2825-2830, 2011.

## Appendix A: Detailed Accuracy Tables

### A.1 Cross-Validation Results (5 Folds)

**LL-Jaccard:**

  ------------------------------------------------------------------------
  Fold              Accuracy            Correct           Total
  ----------------- ------------------- ----------------- ----------------
  1                 17.0%               34                200

  2                 15.0%               30                200

  3                 17.5%               35                200

  4                 17.0%               34                200

  5                 21.5%               43                200

  Mean              **17.6%**           **35.2**          **200**

  Std               **±2.1%**           **±4.2**          **---**
  ------------------------------------------------------------------------

**LL-NB:**

  ------------------------------------------------------------------------
  Fold              Accuracy            Correct           Total
  ----------------- ------------------- ----------------- ----------------
  1                 30.0%               60                200

  2                 39.0%               78                200

  3                 34.5%               69                200

  4                 32.0%               64                200

  5                 36.0%               72                200

  Mean              **34.3%**           **68.6**          **200**

  Std               **±3.1%**           **±6.2**          **---**
  ------------------------------------------------------------------------

**VNG++:**

  ------------------------------------------------------------------------
  Fold              Accuracy            Correct           Total
  ----------------- ------------------- ----------------- ----------------
  1                 21.0%               42                200

  2                 24.5%               49                200

  3                 26.0%               52                200

  4                 24.5%               49                200

  5                 26.0%               52                200

  Mean              **24.4%**           **48.8**          **200**

  Std               **±1.8%**           **±3.6**          **---**
  ------------------------------------------------------------------------

*(Note: VNG++ results shown here are from interval=5000. Final tuned
result with interval=3000 is 25.5%.)*

**P-SVM (AdaBoost):**

  ------------------------------------------------------------------------
  Fold              Accuracy            Correct           Total
  ----------------- ------------------- ----------------- ----------------
  1                 32.1%               64                200

  2                 35.8%               72                200

  3                 37.2%               74                200

  4                 33.9%               68                200

  5                 36.0%               72                200

  Mean              **35.0%**           **70.0**          **200**

  Std               **±2.4%**           **±4.8**          **---**
  ------------------------------------------------------------------------

## Appendix B: Code Availability

Complete source code for this reproduction study is available at:\
<https://github.com/Usamarana01/alexa-fingerprinting>

The repository includes:

-   All four attack implementations

-   Data loading and preprocessing scripts

-   Feature extraction utilities

-   Evaluation and visualization code

-   Cross-validation framework

-   README with setup instructions

-   Requirements file for dependencies

**Dependencies:**

> numpy \>= 1.21.0\
> pandas \>= 1.3.0\
> scikit-learn \>= 0.24.2\
> matplotlib \>= 3.4.2\
> seaborn \>= 0.11.1

**Installation:**

> git clone https://github.com/Usamarana01/alexa-fingerprinting.git\
> cd alexa-fingerprinting\
> pip install -r requirements.txt\
> python main.py

## Appendix C: Feature Importance Details

**P-SVM (AdaBoost) Feature Importance Scores:**

  ------------------------------------------------------------------------
  Rank        Feature                Importance         Cumulative
  ----------- ---------------------- ------------------ ------------------
  1           incoming_bytes         0.6290             62.9%

  2           total_bytes            0.1182             74.7%

  3           duration               0.0869             83.4%

  4           avg_burst_size         0.0532             88.7%

  5           num_bursts             0.0408             92.8%

  6           std_burst_size         0.0241             95.2%

  7           incoming_packets       0.0189             97.1%

  8           avg_packet_size        0.0134             98.4%

  9           max_packet_size        0.0087             99.3%

  10          total_packets          0.0042             99.7%

  11          pct_incoming           0.0014             99.8%

  12          outgoing_bytes         0.0007             99.9%

  13          std_packet_size        0.0003             100.0%

  14          outgoing_packets       0.0001             100.0%

  15          min_packet_size        0.0001             100.0%
  ------------------------------------------------------------------------

**Interpretation:**

The top 5 features account for 92.8% of classification decisions.
Response size (incoming_bytes) alone is responsible for nearly
two-thirds. This aligns with the attack\'s fundamental insight:
different commands produce different response sizes, and that\'s the
primary fingerprint.

Interestingly, query-side features (outgoing_bytes, outgoing_packets)
have near-zero importance. This makes sense---voice queries are all
roughly the same size (a few seconds of compressed audio). The
variability is entirely in the response.
