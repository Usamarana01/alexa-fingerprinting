"""
pcap_to_csv.py — Convert PCAP/PCAPNG to Project CSV Format
=============================================================
If you already captured traffic with Wireshark GUI, use this
script to convert .pcap / .pcapng files to the project format.

Output CSV columns: ,time,size,direction
  direction: +1.0 = outgoing, -1.0 = incoming

Requirements:
    pip install pyshark pandas

Usage:
    python "data collection/pcap_to_csv.py" <input.pcap> <output.csv> [my_ip]

Example:
    python "data collection/pcap_to_csv.py" captures/alexa.pcap data/trace_csv/what_is_weather_5_1.csv 192.168.1.5

    # Convert all PCAPs in a folder:
    python "data collection/pcap_to_csv.py" --batch captures/ data/trace_csv/ 192.168.1.5
"""

import pyshark
import pandas as pd
import sys
import os
import glob


def pcap_to_csv(pcap_file, output_csv, my_ip=None):
    """Convert a single PCAP file to CSV"""
    print(f"\nReading: {pcap_file}")

    if not os.path.exists(pcap_file):
        print(f"  ❌ File not found: {pcap_file}")
        return None

    try:
        cap = pyshark.FileCapture(pcap_file)
    except Exception as e:
        print(f"  ❌ Cannot open file: {e}")
        return None

    packets   = []
    start_time = None

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

            if my_ip:
                direction = 1.0 if src_ip == my_ip else -1.0
            else:
                direction = 1.0  # unknown direction; set all to outgoing

            packets.append({
                'time': relative_time,
                'size': float(pkt_size),
                'direction': direction
            })

        except AttributeError:
            continue

    cap.close()

    if not packets:
        print("  ⚠️  No IP packets found in PCAP")
        return None

    df = pd.DataFrame(packets)

    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    df.to_csv(output_csv)

    print(f"  ✅ {len(df)} packets → {output_csv}")
    print(f"  Duration: {df['time'].max():.2f}s")
    print(f"  Outgoing: {(df['direction'] == 1.0).sum()} packets")
    print(f"  Incoming: {(df['direction'] == -1.0).sum()} packets")
    return df


def batch_convert(input_dir, output_dir, my_ip=None):
    """Convert all PCAP files in a directory"""
    pcap_files = glob.glob(os.path.join(input_dir, "*.pcap")) + \
                 glob.glob(os.path.join(input_dir, "*.pcapng"))

    if not pcap_files:
        print(f"  No PCAP files found in {input_dir}")
        return

    print(f"Found {len(pcap_files)} PCAP files")
    os.makedirs(output_dir, exist_ok=True)

    for pcap_file in pcap_files:
        stem       = os.path.splitext(os.path.basename(pcap_file))[0]
        output_csv = os.path.join(output_dir, f"{stem}.csv")
        pcap_to_csv(pcap_file, output_csv, my_ip)


def print_usage():
    print(__doc__)


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    # batch mode
    if sys.argv[1] == "--batch":
        if len(sys.argv) < 4:
            print("Usage: python pcap_to_csv.py --batch <input_dir> <output_dir> [my_ip]")
            sys.exit(1)
        input_dir  = sys.argv[2]
        output_dir = sys.argv[3]
        my_ip      = sys.argv[4] if len(sys.argv) > 4 else None
        batch_convert(input_dir, output_dir, my_ip)

    else:
        if len(sys.argv) < 3:
            print("Usage: python pcap_to_csv.py <input.pcap> <output.csv> [my_ip]")
            sys.exit(1)
        pcap_file  = sys.argv[1]
        output_csv = sys.argv[2]
        my_ip      = sys.argv[3] if len(sys.argv) > 3 else None
        pcap_to_csv(pcap_file, output_csv, my_ip)


if __name__ == "__main__":
    main()
