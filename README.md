# Encrypted Traffic Fingerprinting: Alexa & Websites

This project implements traffic analysis attacks to fingerprint encrypted network traffic. It reproduces the findings of "I Can Hear Your Alexa" (IEEE CNS 2019) and extends the methodology to modern website fingerprinting (2026).

**Goal**: Identify which voice command was issued to an Amazon Echo or which website was visited on a browser by analyzing **only** encrypted traffic metadata (packet sizes, timing, direction).

## 🚀 Quick Start

### 1. Prerequisites
-   **Python 3.8+**
-   **Wireshark (with Tshark)** must be installed and in your PATH (or configured in scripts).
    -   *Windows*: Usually at `C:\Program Files\Wireshark\tshark.exe`
-   **Npcap** (installed with Wireshark) for packet capturing on Windows.

### 2. Installation
Clone the repo and install dependencies:

```bash
pip install -r requirements.txt
```

*(Key libraries: `scikit-learn`, `pandas`, `numpy`, `pyshark`, `matplotlib`, `seaborn`)*

---

## 📡 Data Collection

We support two types of data collection.

### A. Website Fingerprinting (Browser)
Collects traffic while you visit websites on your computer.

1.  **Run the collector:**
    ```bash
    # Run as Administrator (required for packet capture)
    python "data collection/collect_websites.py"
    ```
2.  **Follow the prompts:** The script will ask you to open specific URLs (Google, YouTube, etc.) and will automatically record the traffic.
3.  **Output:**
    -   Live data appended to: `data collection/collection.csv`
    -   Individual traces saved to: `data/trace_csv/`

### B. Voice Command Fingerprinting (Alexa)
Collects traffic from an Alexa device on your network.

1.  **Configure IP:** Open `data collection/collect_traffic.py` and set:
    -   `TARGET_IP`: The local IP address of your Amazon Echo.
2.  **Run the collector:**
    ```bash
    python "data collection/collect_traffic.py"
    ```
3.  **Interact:** Speak the prompted commands to Alexa.

---

## 🧠 Running the Analysis

Once data is collected (or using the pre-existing dataset), run the main analysis pipeline to train models and evaluate attack accuracy.

```bash
python main.py
```

This will:
1.  Load traces from `data/trace_csv/`.
2.  Extract features (Packet histograms, Burst sizes, etc.).
3.  Train four classifiers:
    -   **LL-Jaccard**: Simple set-similarity.
    -   **LL-NB**: Naive Bayes on packet size histograms.
    -   **VNG++**: Naive Bayes on burst size histograms.
    -   **P-SVM**: AdaBoost with statistical features (State-of-the-Art).
4.  Generate results and confusion matrices in the `results/` folder.

---

## 📂 Project Structure

```
├── data/
│   └── trace_csv/          # Raw CSV packet traces (timestamp, size, direction)
├── data collection/
│   ├── collect_websites.py # Script for browser traffic collection
│   ├── collect_traffic.py  # Script for Alexa traffic collection
│   └── collection.csv      # Centralized log of all collected data
├── src/
│   ├── attacks/            # Implementation of fingerprinting algorithms
│   ├── data_loader.py      # Trace parsing and dataset splitting
│   └── feature_extraction.py # Feature engineering logic
├── results/                # Generated figures and accuracy tables
├── main.py                 # Entry point for training and evaluation
└── requirements.txt        # Python dependencies
```

