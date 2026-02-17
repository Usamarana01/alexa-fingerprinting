"""
Feature Extraction Module
Implements feature extraction matching the paper's reference code

Key insight from reference code:
- Features are computed as size * direction (signed size)
- Bayes: histogram bins over signed packet sizes
- VNG++: histogram bins over signed burst sizes + statistics
- SVM: burst histogram + 5 statistical features
- Jaccard: set of signed packet sizes (size * direction)
"""

import numpy as np
from typing import List, Tuple, Set
import math


class FeatureExtractor:
    """
    Extract features from traffic traces for fingerprinting attacks
    """

    @staticmethod
    def extract_ll_features(trace: List[Tuple[float, int, int]],
                           rounding: int = None) -> Set[int]:
        """
        Extract features for LL-Jaccard attack

        From reference code (trainByJaccard.readfile):
        Each packet -> size * direction (signed size)
        Returns set of unique signed sizes.

        Args:
            trace: List of (timestamp, size, direction) tuples
            rounding: Not used for Jaccard (kept for API compatibility)

        Returns:
            Set of signed packet sizes (size * direction)
        """
        features = set()

        for timestamp, size, direction in trace:
            # Reference: elem = str2int(tmp[-1]) * str2int(tmp[-2])
            # i.e., direction * size
            signed_size = int(size * direction)
            features.add(signed_size)

        return features

    @staticmethod
    def extract_bursts(trace: List[Tuple[float, int, int]]) -> List[int]:
        """
        Extract traffic bursts following reference code (trainByVNGpp.calculateBursts)

        Burst = sum of (size * direction) for consecutive same-direction packets

        Args:
            trace: List of (timestamp, size, direction) tuples

        Returns:
            List of signed burst sizes
        """
        if not trace:
            return []

        burst_list = []

        # First packet initializes
        direction = trace[0][2]
        tmp_burst = int(trace[0][1] * trace[0][2])  # size * direction

        for i in range(1, len(trace)):
            timestamp, size, cur_direction = trace[i]
            cur_direction = int(cur_direction)

            if cur_direction != direction:
                burst_list.append(tmp_burst)
                direction = cur_direction
                tmp_burst = int(size * cur_direction)
            else:
                tmp_burst += int(size * cur_direction)

        # Add final burst
        burst_list.append(tmp_burst)

        return burst_list

    @staticmethod
    def _get_section_list(start: int, end: int, interval: int):
        """
        Create histogram bins following reference code (tools.getSectionList)

        Creates range boundaries and section counts for histogram-based features.

        Args:
            start: Start of range
            end: End of range (exclusive)
            interval: Bin width

        Returns:
            range_list: List of bin boundaries
            section_list: List of zeros (counts)
        """
        range_list = list(range(start, end, interval))
        # +2 for underflow and overflow bins
        section_list = [0] * (len(range_list) + 1)
        return range_list, section_list

    @staticmethod
    def _compute_range(range_list: List[int], value: int) -> int:
        """
        Find which histogram bin a value falls into
        following reference code (tools.computeRange)

        Args:
            range_list: Sorted bin boundaries
            value: Value to bin

        Returns:
            Bin index
        """
        for i in range(len(range_list)):
            if value < range_list[i]:
                return i
        return len(range_list)

    @staticmethod
    def extract_bayes_features(trace: List[Tuple[float, int, int]],
                               interval: int = 100) -> List[int]:
        """
        Extract histogram features for LL-NB (Bayes) attack

        From reference code (trainByBayes.computeFeature):
        1. Compute signed size (size * direction) for each packet
        2. Bin into histogram with range [-1500, 1501, interval]

        Args:
            trace: List of (timestamp, size, direction) tuples
            interval: Histogram bin width (default 100, paper tests 10-100)

        Returns:
            Histogram feature vector
        """
        start, end = -1500, 1501
        range_list, section_list = FeatureExtractor._get_section_list(start, end, interval)

        for timestamp, size, direction in trace:
            signed_size = int(size * direction)
            index = FeatureExtractor._compute_range(range_list, signed_size)
            section_list[index] += 1

        return section_list

    @staticmethod
    def extract_vng_features(trace: List[Tuple[float, int, int]],
                            interval: int = 5000) -> List[float]:
        """
        Extract features for VNG++ attack

        From reference code (trainByVNGpp.computeFeature):
        1. Read file to get statistics and tuple list
        2. Calculate bursts (signed burst sizes)
        3. Bin bursts into histogram [-400000, 400001, interval]
        4. Prepend [TotalTraceTime, UpStreamTotal, DownStreamTotal]

        Args:
            trace: List of (timestamp, size, direction) tuples
            interval: Burst histogram bin width (default 5000)

        Returns:
            Feature vector: [trace_time, up_bytes, down_bytes] + burst_histogram
        """
        if not trace:
            return [0, 0, 0] + [0] * 162  # default empty

        # Compute statistics (like trainByVNGpp.readfile)
        up_stream_total = 0
        down_stream_total = 0
        timestamps = []

        for timestamp, size, direction in trace:
            timestamps.append(timestamp)
            if direction == 1:
                up_stream_total += int(size)
            elif direction == -1:
                down_stream_total += int(size)

        # Total trace time
        timestamps.sort()
        total_trace_time = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0

        # Calculate bursts
        burst_list = FeatureExtractor.extract_bursts(trace)

        # Bin bursts into histogram
        start, end = -400000, 400001
        range_list, section_list = FeatureExtractor._get_section_list(start, end, interval)

        for burst in burst_list:
            index = FeatureExtractor._compute_range(range_list, burst)
            section_list[index] += 1

        # Compose feature vector: [time, up_bytes, down_bytes] + histogram
        features = [total_trace_time, up_stream_total, down_stream_total]
        features.extend(section_list)

        return features

    @staticmethod
    def extract_svm_features(trace: List[Tuple[float, int, int]],
                            interval: int = 5000) -> List[float]:
        """
        Extract features for SVM attack

        From reference code (trainBySvm.computeFeature):
        1. Get packet statistics from readfile
        2. Calculate bursts
        3. Bin bursts into histogram [-200000, 200001, interval]
        4. Compute: upStreamTotal, downStreamTotal, inPackRatio, packNum, burstNum
        5. Combine: [stats] + histogram

        Args:
            trace: List of (timestamp, size, direction) tuples
            interval: Burst histogram bin width

        Returns:
            Feature vector: [up_bytes, down_bytes, in_ratio, pack_num, burst_num] + burst_histogram
        """
        if not trace:
            return [0] * 86  # default empty

        # Compute statistics
        up_pack_num = 0
        down_pack_num = 0
        up_stream_total = 0
        down_stream_total = 0

        for timestamp, size, direction in trace:
            if direction == 1:
                up_stream_total += int(size)
                up_pack_num += 1
            elif direction == -1:
                down_stream_total += int(size)
                down_pack_num += 1

        # Calculate bursts
        burst_list = FeatureExtractor.extract_bursts(trace)

        # Bin bursts
        start, end = -200000, 200001
        range_list, section_list = FeatureExtractor._get_section_list(start, end, interval)

        for burst in burst_list:
            index = FeatureExtractor._compute_range(range_list, burst)
            section_list[index] += 1

        # Statistics
        total_packs = up_pack_num + down_pack_num
        in_pack_ratio = down_pack_num / total_packs if total_packs > 0 else 0
        pack_num = len(trace)
        burst_num = len(burst_list)

        features = [up_stream_total, down_stream_total, in_pack_ratio, pack_num, burst_num]
        features.extend(section_list)

        return features

    @staticmethod
    def extract_all_features(trace: List[Tuple[float, int, int]]) -> dict:
        """
        Extract all feature types for a trace

        Args:
            trace: List of (timestamp, size, direction) tuples

        Returns:
            Dictionary with all feature types
        """
        return {
            'll_features': FeatureExtractor.extract_ll_features(trace),
            'bayes_features': FeatureExtractor.extract_bayes_features(trace, interval=100),
            'vng_features': FeatureExtractor.extract_vng_features(trace, interval=5000),
            'svm_features': FeatureExtractor.extract_svm_features(trace, interval=5000),
            'bursts': FeatureExtractor.extract_bursts(trace)
        }


# Test code
if __name__ == "__main__":
    sample_trace = [
        (0.000, 150, 1),
        (0.023, 80, 1),
        (0.045, 60, 1),
        (0.100, 1200, -1),
        (0.123, 800, -1),
        (0.145, 600, -1),
        (0.200, 100, 1),
        (0.220, 50, 1),
    ]

    print("Sample Trace:")
    for t, s, d in sample_trace:
        print(f"  t={t:.3f}, size={s}, dir={'out' if d==1 else 'in'}")
    print()

    # Test LL features (Jaccard)
    ll_features = FeatureExtractor.extract_ll_features(sample_trace)
    print(f"LL Features (signed sizes): {sorted(ll_features)}")
    print()

    # Test bursts
    bursts = FeatureExtractor.extract_bursts(sample_trace)
    print(f"Bursts (signed): {bursts}")
    print()

    # Test Bayes features
    bayes = FeatureExtractor.extract_bayes_features(sample_trace, interval=100)
    print(f"Bayes histogram ({len(bayes)} bins), non-zero:")
    for i, v in enumerate(bayes):
        if v > 0:
            print(f"  bin[{i}] = {v}")
    print()

    # Test VNG features
    vng = FeatureExtractor.extract_vng_features(sample_trace, interval=5000)
    print(f"VNG features ({len(vng)} dims)")
    print(f"  TraceTime={vng[0]:.3f}, UpBytes={vng[1]}, DownBytes={vng[2]}")
    print()

    # Test SVM features
    svm = FeatureExtractor.extract_svm_features(sample_trace, interval=5000)
    print(f"SVM features ({len(svm)} dims)")
    print(f"  UpBytes={svm[0]}, DownBytes={svm[1]}, InRatio={svm[2]:.2f}, "
          f"Packs={svm[3]}, Bursts={svm[4]}")
