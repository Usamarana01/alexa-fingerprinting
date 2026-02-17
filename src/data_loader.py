"""
Data Loader Module
Loads and preprocesses traffic traces from the trace_csv dataset

Actual data format:
  - Flat directory: data/trace_csv/
  - 1000 CSV files with columns: (unnamed index), time, size, direction
  - File naming: {command_name}_{variant}_{trace_id}.csv
  - Each command has ~10 traces (5 regular + 5 with _L_ suffix)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import re
from collections import defaultdict


class TrafficDataLoader:
    """
    Loads traffic traces from flat CSV directory structure

    Expected file naming patterns:
        what_time_is_it_5_L_1.csv
        what_time_is_it_5_0_18_31_47_58_capture_1.csv
        alexa_5_30s_L_1.csv
        tell_me_a_joke_5_30s_1.csv
    """

    def __init__(self, data_dir: str = 'data/trace_csv'):
        """
        Initialize data loader

        Args:
            data_dir: Path to directory containing CSV trace files
        """
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        print(f"DataLoader initialized with: {self.data_dir}")

    @staticmethod
    def extract_command_name(filename: str) -> str:
        """
        Extract command name from CSV filename.

        Strips the trailing trace ID and variant info to get the base command.
        Pattern: the last _N.csv is the trace number.
        We group all traces of the same command together.

        Examples:
            'alexa_5_30s_L_1.csv' -> 'alexa'
            'what_time_is_it_5_L_3.csv' -> 'what_time_is_it'
            'tell_me_a_joke_5_30s_2.csv' -> 'tell_me_a_joke'
            'what_time_is_it_5_0_18_31_47_58_capture_3.csv' -> 'what_time_is_it'

        Strategy: find the '_5_' separator which appears in all filenames
        after the command name (it indicates 5 repetitions).
        """
        name = filename.replace('.csv', '')

        # Find the FIRST occurrence of '_5_' — everything before it is the command
        idx = name.find('_5_')
        if idx != -1:
            return name[:idx]

        # Fallback: try splitting on last _N pattern
        match = re.match(r'^(.+?)_\d+$', name)
        if match:
            return match.group(1)

        return name

    def load_trace(self, trace_file: Path) -> List[Tuple[float, int, int]]:
        """
        Load single traffic trace from CSV file

        Args:
            trace_file: Path to CSV file

        Returns:
            List of (timestamp, size, direction) tuples
        """
        try:
            df = pd.read_csv(trace_file)

            # The actual columns are: unnamed index, time, size, direction
            # Identify the right column names
            if 'time' in df.columns and 'size' in df.columns and 'direction' in df.columns:
                time_col, size_col, dir_col = 'time', 'size', 'direction'
            elif 'timestamp' in df.columns and 'packet_length' in df.columns:
                time_col, size_col, dir_col = 'timestamp', 'packet_length', 'direction'
            else:
                # Try positional (skip index column if present)
                cols = df.columns.tolist()
                if len(cols) >= 3:
                    # If first column looks like an index, skip it
                    start = 1 if cols[0] == '' or cols[0] == 'Unnamed: 0' else 0
                    time_col = cols[start]
                    size_col = cols[start + 1]
                    dir_col = cols[start + 2]
                else:
                    raise ValueError(f"Cannot identify columns in {trace_file}: {cols}")

            trace = []
            for _, row in df.iterrows():
                timestamp = float(row[time_col])
                size = int(abs(float(row[size_col])))
                direction = int(float(row[dir_col]))

                # Normalize direction to 1 (outgoing) or -1 (incoming)
                if direction > 0:
                    direction = 1
                elif direction < 0:
                    direction = -1
                else:
                    continue  # skip zero-direction packets

                if size <= 0:
                    continue

                trace.append((timestamp, size, direction))

            return trace

        except Exception as e:
            print(f"Error loading {trace_file}: {e}")
            return []

    def load_all_traces(self, max_commands: int = None) -> Tuple[List, np.ndarray, List[str]]:
        """
        Load all traffic traces from dataset

        Args:
            max_commands: Maximum number of commands to load (None = all)

        Returns:
            X: List of traffic traces
            y: Array of labels (command indices)
            command_names: List of command names
        """
        # Group files by command name
        command_files: Dict[str, List[Path]] = defaultdict(list)

        csv_files = sorted(self.data_dir.glob('*.csv'))
        print(f"Found {len(csv_files)} CSV files")

        for csv_file in csv_files:
            cmd_name = self.extract_command_name(csv_file.name)
            command_files[cmd_name].append(csv_file)

        # Sort commands alphabetically
        sorted_commands = sorted(command_files.keys())

        if max_commands:
            sorted_commands = sorted_commands[:max_commands]

        print(f"Loading traces from {len(sorted_commands)} commands...")

        X = []
        y = []
        command_names = []

        for cmd_idx, cmd_name in enumerate(sorted_commands):
            files = command_files[cmd_name]
            command_names.append(cmd_name)

            loaded_count = 0
            for trace_file in files:
                trace = self.load_trace(trace_file)

                if len(trace) > 0:
                    X.append(trace)
                    y.append(cmd_idx)
                    loaded_count += 1

            if (cmd_idx + 1) % 20 == 0 or cmd_idx == 0:
                print(f"  [{cmd_idx+1}/{len(sorted_commands)}] {cmd_name}: {loaded_count} traces")

        print(f"\nLoaded {len(X)} traces for {len(command_names)} commands")

        return X, np.array(y), command_names

    def split_train_test(self, X: List, y: np.ndarray,
                         train_ratio: float = 0.8,
                         random_state: int = 42) -> Tuple:
        """
        Split data into train/test sets (stratified)

        Args:
            X: List of traces
            y: Array of labels
            train_ratio: Ratio of training data (default: 0.8 = 80%)
            random_state: Random seed for reproducibility

        Returns:
            X_train, X_test, y_train, y_test
        """
        from sklearn.model_selection import train_test_split

        # Check if there are enough samples for stratification
        unique, counts = np.unique(y, return_counts=True)
        min_count = counts.min()

        if min_count >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                train_size=train_ratio,
                stratify=y,
                random_state=random_state
            )
        else:
            # Fallback: no stratification if some classes have only 1 sample
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                train_size=train_ratio,
                random_state=random_state
            )
            print("  Warning: Some classes have too few samples for stratification")

        print(f"\nDataset split:")
        print(f"  Training: {len(X_train)} traces")
        print(f"  Testing:  {len(X_test)} traces")

        return X_train, X_test, y_train, y_test

    def get_statistics(self, X: List, y: np.ndarray) -> dict:
        """
        Calculate dataset statistics

        Args:
            X: List of traces
            y: Array of labels

        Returns:
            Dictionary with statistics
        """
        trace_lengths = [len(trace) for trace in X]

        stats = {
            'total_traces': len(X),
            'num_commands': len(np.unique(y)),
            'avg_packets_per_trace': np.mean(trace_lengths),
            'min_packets': min(trace_lengths),
            'max_packets': max(trace_lengths),
        }

        # Calculate average trace duration
        durations = []
        for trace in X:
            if len(trace) > 1:
                duration = trace[-1][0] - trace[0][0]
                durations.append(duration)

        stats['avg_duration'] = np.mean(durations) if durations else 0

        return stats


# Test code
if __name__ == "__main__":
    loader = TrafficDataLoader('data/trace_csv')

    # Load all data
    X, y, command_names = loader.load_all_traces()

    # Print statistics
    stats = loader.get_statistics(X, y)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Show commands
    print(f"\nCommands ({len(command_names)}):")
    for i, name in enumerate(command_names):
        count = np.sum(y == i)
        print(f"  [{i:3d}] {name} ({count} traces)")

    # Split data
    X_train, X_test, y_train, y_test = loader.split_train_test(X, y)

    # Show first trace sample
    print(f"\nFirst trace from '{command_names[0]}':")
    print(f"  Packets: {len(X[0])}")
    print(f"  First 5 packets:")
    for i in range(min(5, len(X[0]))):
        t, s, d = X[0][i]
        print(f"    {i+1}. t={t:.3f}s, size={s}B, dir={'out' if d==1 else 'in'}")
