"""
collect_websites.py — Website Fingerprinting Data Collection
============================================================
Captures HTTPS traffic when visiting websites in your browser.
No Alexa device required — just a browser and this script.

The captured data format is identical to the Alexa dataset:
    columns: ,time,size,direction

Requirements:
    pip install pyshark pandas
    Wireshark + npcap must be installed from wireshark.org

Usage:
    1. Edit CONFIG section below
    2. Run: python "data collection/collect_websites.py"
    3. When prompted, open your browser and visit the website
    4. Close all other browser tabs before starting

Author: Data Collection Script for Alexa Fingerprinting Project
"""

import pyshark
import pandas as pd
import time
import os
import socket

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TSHARK_PATH    = r"C:\Program Files\Wireshark\tshark.exe"    # Path to tshark.exe
INTERFACE      = r"\Device\NPF_{4B920F42-F4E8-4FE4-AE3A-2BE7C3E7B1C7}"      # Your network adapter name (run: tshark -D)
MY_IP          = ""           # Leave empty for auto-detection
CAPTURE_SECS   = 15           # Seconds to capture per website visit
OUTPUT_DIR     = "data/trace_csv"
NUM_REPEATS    = 5            # How many times to visit each site

# Websites to fingerprint (top-50 style list)
WEBSITES = [
    "google.com",
    "youtube.com",
    "amazon.com",
    "wikipedia.org",
    "reddit.com",
    "github.com",
    "twitter.com",
    "facebook.com",
    "netflix.com",
    "instagram.com",
    "linkedin.com",
    "yahoo.com",
    "bing.com",
    "microsoft.com",
    "apple.com",
    "stackoverflow.com",
    "twitch.tv",
    "pinterest.com",
    "wordpress.com",
    "quora.com",
    "bbc.com",
    "cnn.com",
    "nytimes.com",
    "theguardian.com",
    "ebay.com",
]

# ─────────────────────────────────────────────────────────────────────────────


def get_my_ip():
    """Auto-detect local IP"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def capture_website_trace(website, repeat, my_ip):
    """Capture traffic for a single website visit with real-time logging"""
    print(f"\n  ⏳ Capturing '{website}' for {CAPTURE_SECS}s...")
    print(f"  ➡️  Quickly open your browser and go to: https://{website}")
    print(f"     (Make sure to close other tabs!)")

    packets = []
    start_time = None
    
    # Configure tshark path
    import pyshark.config
    try:
        pyshark.config.get_config().tshark_path = TSHARK_PATH
    except Exception:
        pass # Handle cases where config might not be available

    try:
        cap = pyshark.LiveCapture(
            interface=INTERFACE,
            bpf_filter="ip"   # Capture all IP traffic
        )
        
        # We use sniff_continuously to show real-time progress
        print(f"  📡 Listening for packets... (0 captured)", end='\r')
        
        start_sniff = time.time()
        for pkt in cap.sniff_continuously():
            if time.time() - start_sniff > CAPTURE_SECS:
                break
                
            try:
                pkt_time = float(pkt.sniff_timestamp)
                pkt_size = int(pkt.length)

                if not hasattr(pkt, 'ip'):
                    continue

                src_ip = pkt.ip.src

                if start_time is None:
                    start_time = pkt_time

                relative_time = round(pkt_time - start_time, 10)
                direction = 1.0 if src_ip == my_ip else -1.0

                packets.append({
                    'time': relative_time,
                    'size': float(pkt_size),
                    'direction': direction
                })
                
                # Real-time progress update
                if len(packets) % 2 == 0:
                    elapsed = time.time() - start_sniff
                    print(f"  📡 Listening for packets... ({len(packets)} captured, {elapsed:.1f}s elapsed)", end='\r')

            except AttributeError:
                continue
        
        print(f"\n  ✅ Sniffing done.")

    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        return None

    if len(packets) < 5:
        print(f"  ⚠️  Too few packets ({len(packets)})")
        return None

    df = pd.DataFrame(packets)
    print(f"  📊 Final count: {len(df)} packets, duration={df['time'].max():.2f}s")
    return df


def save_trace(df, site_name, repeat):
    """Save to CSV in the project format and append to central collection.csv"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_name = site_name.replace(".", "_").replace("-", "_")
    filename  = f"{safe_name}_5_{repeat}.csv"
    filepath  = os.path.join(OUTPUT_DIR, filename)
    
    # 1. Save individual trace file (for main.py compatibility)
    df.to_csv(filepath)
    print(f"  💾 Saved individual: {filename}")

    # 2. Append to central collection.csv
    central_file = os.path.join("data collection", "collection.csv")
    
    # Prepare data for appending: add 'label' column
    label = f"{safe_name}_repeat_{repeat}"
    append_df = df.copy()
    append_df.insert(0, 'label', label)
    
    # We don't want the index in the central file, and we don't want the header if file already exists
    header = not os.path.exists(central_file) or os.path.getsize(central_file) == 0
    append_df.to_csv(central_file, mode='a', index=False, header=header)
    print(f"  📝 Appended to: collection.csv")
    
    return filepath


def main():
    my_ip = MY_IP if MY_IP else get_my_ip()

    print(f"\n{'='*55}")
    print(f"  Website Fingerprinting — Traffic Collection")
    print(f"{'='*55}")
    print(f"  My IP     : {my_ip}")
    print(f"  Interface : {INTERFACE}")
    print(f"  Sites     : {len(WEBSITES)}")
    print(f"  Repeats   : {NUM_REPEATS}")
    print(f"  Per visit : {CAPTURE_SECS}s")
    print(f"  Total     : {len(WEBSITES) * NUM_REPEATS} captures")
    print(f"{'='*55}")
    print("\n  TIP: Close all browser tabs before each capture.")
    print("       Wait for the page to fully load during each capture.\n")

    total = len(WEBSITES) * NUM_REPEATS
    done  = 0
    saved = 0

    for site in WEBSITES:
        for repeat in range(1, NUM_REPEATS + 1):
            done += 1
            print(f"\n[{done}/{total}] Site: {site} | Repeat: #{repeat}")
            input(f"  Press ENTER, then immediately open: https://{site}")

            df = capture_website_trace(site, repeat, my_ip)

            if df is not None and len(df) >= 5:
                save_trace(df, site, repeat)
                saved += 1
            else:
                print(f"  ⏭️  Skipping this capture")

            time.sleep(3)  # Pause between captures to let connections close

    print(f"\n{'='*55}")
    print(f"  ✅ Done!")
    print(f"  Saved: {saved}/{total} traces")
    print(f"  Folder: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
