"""
collect_traffic.py — Automated Traffic Data Collection
=======================================================
Captures encrypted traffic traces for voice/smart-speaker fingerprinting.
Saves CSV files in the format used by this project:
    columns: ,time,size,direction
    direction: +1.0 = outgoing (your laptop → target)
               -1.0 = incoming (target → your laptop)

Requirements:
    pip install pyshark pandas
    Wireshark + npcap must be installed from wireshark.org

Usage:
    1. Edit the CONFIG section below
    2. Run: python "data collection/collect_traffic.py"
    3. Follow the on-screen prompts

Author: Data Collection Script for Alexa Fingerprinting Project
"""

import pyshark
import pandas as pd
import time
import os
import socket
import sys

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — Edit these values before running
# ─────────────────────────────────────────────────────────────────────────────

INTERFACE = r"\Device\NPF_{4B920F42-F4E8-4FE4-AE3A-2BE7C3E7B1C7}"          # Run 'tshark -D' in CMD to see your interface names
TARGET_IP = ""               # IP of Alexa/smart speaker (leave empty to capture all traffic)
MY_IP     = ""               # Your laptop IP (leave empty for auto-detection)

CAPTURE_SECONDS = 10         # How long to capture per voice command (seconds)
OUTPUT_DIR      = "data/trace_csv"
NUM_REPEATS     = 5          # How many captures per command (paper uses 5)

# Voice commands to capture (will be used as part of the filename)
COMMANDS = [
    "what_is_the_weather",
    "tell_me_a_joke",
    "play_music",
    "set_a_timer_for_thirty_seconds",
    "how_old_is_lebron_james",
    "what_time_is_it",
    "good_morning",
    "give_me_a_fun_fact",
    "flip_a_coin",
    "tell_me_a_story",
]

# ─────────────────────────────────────────────────────────────────────────────


def get_my_ip():
    """Auto-detect our local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def capture_trace(command_name, capture_index, my_ip, target_ip):
    """
    Capture a single traffic trace for one voice command.
    Returns a DataFrame with columns: time, size, direction
    """
    print(f"\n{'─'*55}")
    print(f"  Command : '{command_name.replace('_', ' ')}'")
    print(f"  Repeat  : #{capture_index} of {NUM_REPEATS}")
    print(f"  Duration: {CAPTURE_SECONDS}s")
    print(f"{'─'*55}")
    print(f"  ➡️  SAY THE COMMAND TO YOUR ALEXA/DEVICE NOW")
    print(f"  ⏳ Capturing...")

    # Build filter
    if target_ip:
        bpf_filter = f"host {target_ip}"
    else:
        bpf_filter = "ip"  # Capture all IP traffic

    packets = []
    start_time = None

    try:
        cap = pyshark.LiveCapture(
            interface=INTERFACE,
            bpf_filter=bpf_filter
        )
        cap.sniff(timeout=CAPTURE_SECONDS)

        for pkt in cap:
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

            except AttributeError:
                continue

    except Exception as e:
        print(f"\n  ❌ Capture error: {e}")
        print("  → Check: Is the interface name correct? Are you running as Admin?")
        return None

    if len(packets) < 5:
        print(f"  ⚠️  Only {len(packets)} packets captured — try again")
        return None

    df = pd.DataFrame(packets)
    duration = df['time'].max()
    print(f"  ✅ Captured {len(df)} packets  |  Duration: {duration:.2f}s")
    return df


def save_trace(df, command_name, capture_index):
    """Save trace DataFrame to CSV and append to central collection.csv"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{command_name}_5_{capture_index}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # 1. Save individual trace file (for main.py compatibility)
    df.to_csv(filepath)
    print(f"  💾 Saved individual: {filepath} ({os.path.getsize(filepath)} bytes)")

    # 2. Append to central collection.csv
    central_file = os.path.join("data collection", "collection.csv")
    
    # Prepare data for appending: add 'label' column
    label = f"{command_name}_repeat_{capture_index}"
    append_df = df.copy()
    append_df.insert(0, 'label', label)
    
    # Append to central file
    header = not os.path.exists(central_file) or os.path.getsize(central_file) == 0
    append_df.to_csv(central_file, mode='a', index=False, header=header)
    print(f"  📝 Appended to: collection.csv")
    
    return filepath


def print_interface_help():
    """Print help for finding interface name"""
    print("\n  To find your interface name, open CMD and run:")
    print("    tshark -D")
    print("  Common names: 'Wi-Fi', 'Ethernet', '\\Device\\NPF_{...}'")


def main():
    # ── Resolve IPs ─────────────────────────────────────────────────────────
    my_ip = MY_IP if MY_IP else get_my_ip()

    print(f"\n{'='*55}")
    print(f"  Traffic Collection — Voice Command Fingerprinting")
    print(f"{'='*55}")
    print(f"  My IP      : {my_ip}")
    print(f"  Target IP  : {TARGET_IP if TARGET_IP else 'All devices (no filter)'}")
    print(f"  Interface  : {INTERFACE}")
    print(f"  Output dir : {OUTPUT_DIR}")
    print(f"  Commands   : {len(COMMANDS)}")
    print(f"  Repeats    : {NUM_REPEATS}")
    print(f"  Total files: {len(COMMANDS) * NUM_REPEATS}")
    print(f"{'='*55}")

    if not TARGET_IP:
        print("\n  ⚠️  TARGET_IP is not set. Capturing ALL traffic.")
        print("     This may include background noise. Setting TARGET_IP")
        print("     to your Alexa's IP gives cleaner data.")

    total = len(COMMANDS) * NUM_REPEATS
    done  = 0
    saved = 0

    for command in COMMANDS:
        for repeat in range(1, NUM_REPEATS + 1):
            done += 1
            print(f"\n[{done}/{total}] Next capture:")
            input(f"  Press ENTER when ready, then say: '{command.replace('_', ' ')}'")

            df = capture_trace(command, repeat, my_ip, TARGET_IP)

            if df is not None and len(df) >= 5:
                save_trace(df, command, repeat)
                saved += 1
            else:
                retry = input("  Retry this capture? (y/n): ").strip().lower()
                if retry == 'y':
                    df = capture_trace(command, repeat, my_ip, TARGET_IP)
                    if df is not None:
                        save_trace(df, command, repeat)
                        saved += 1

            time.sleep(2)  # Short pause between captures

    print(f"\n{'='*55}")
    print(f"  ✅ Collection complete!")
    print(f"  Files saved: {saved}/{total}")
    print(f"  Location: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*55}")


if __name__ == "__main__":
    print_interface_help()
    print()
    main()
