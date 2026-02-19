# Data Collection Report: Website Fingerprinting

## 1. Objective
The goal of this data collection was to gather encrypted network traffic traces from popular websites to train a machine learning model for website fingerprinting. By analyzing the metadata of these traces (specifically packet sizes, timing, and direction), we aim to identify which website a user is visiting, even when the traffic is encrypted (HTTPS/TLS).

## 2. Methodology

### 2.1 Environment
-   **Device**: Windows Laptop (simulating a standard user victim).
-   **Network**: Wi-Fi interface (captured via generic network adapter).
-   **Browser**: Modern web browser (Chrome/Edge).
-   **Tooling**: Python script using `pyshark` (a wrapper for Wireshark's `tshark`).

### 2.2 Collection Process
We developed an automated Python script (`collect_websites.py`) to streamline the collection process:
1.  **Target List**: The script iterates through a predefined list of 25 popular websites (e.g., Google, YouTube, Amazon, Wikipedia).
2.  **User Prompt**: For each website, the script prompts the user to open the specific URL in their browser.
3.  **Traffic Capture**:
    *   The script actively sniffs packets on the network interface for a fixed duration (15 seconds).
    *   It filters for relevant IP traffic (TCP/UDP).
4.  **Repetitions**: Each website was visited **5 times** to account for improved statistical robustness and variability in network conditions.

### 2.3 Data Features
We do **not** decrypt the traffic or inspect the payload. We only interpret packet metadata, which is sufficient for fingerprinting:
-   **Time**: The relative timestamp of when the packet arrived.
-   **Size**: The size of the packet in bytes.
-   **Direction**:
    *   `+1`: Outgoing (uploaded/request)
    *   `-1`: Incoming (downloaded/response)

## 3. Dataset Summary

-   **Total Traces Collected**: 125
-   **Websites**: 25
-   **Traces per Website**: 5
-   **Total Packets**: ~250,000+ (estimated based on average traffic)
-   **Collection Duration**: ~30-45 minutes

## 4. Data Storage
The data is stored in two formats to support different analysis pipelines:
1.  **Individual CSVs**: Saved in `data/trace_csv/` (e.g., `google_com_5_1.csv`).
    *   Format suitable for legacy analysis tools.
2.  **Aggregated Dataset**: Appended to `data collection/collection.csv`.
    *   Format: `label, time, size, direction`
    *   Allows for easy loading into `pandas` for bulk training.

## 5. Conclusion
We have successfully built a robust dataset of encrypted website traffic. This dataset effectively captures the unique "shape" of traffic for each site, providing a solid foundation for training and evaluating our fingerprinting attacks.
