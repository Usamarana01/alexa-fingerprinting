"""
Unit tests for feature extraction
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extraction import FeatureExtractor


class TestFeatureExtractor:

    def setup_method(self):
        """Setup test data"""
        self.simple_trace = [
            (0.0, 150, 1),
            (0.1, 1200, -1),
            (0.2, 80, 1)
        ]

        self.burst_trace = [
            (0.0, 100, 1),
            (0.1, 50, 1),
            (0.2, 200, -1),
            (0.3, 300, -1),
            (0.4, 75, 1)
        ]

    def test_ll_features(self):
        """Test LL features = set of signed sizes (size * direction)"""
        features = FeatureExtractor.extract_ll_features(self.simple_trace)
        # 150*1=150, 1200*(-1)=-1200, 80*1=80
        expected = {150, -1200, 80}
        assert features == expected

    def test_ll_features_signs(self):
        """Test that direction is applied correctly"""
        trace = [(0.0, 500, 1), (0.1, 500, -1)]
        features = FeatureExtractor.extract_ll_features(trace)
        assert 500 in features
        assert -500 in features

    def test_bursts_extraction(self):
        """Test burst extraction (signed bursts)"""
        bursts = FeatureExtractor.extract_bursts(self.burst_trace)
        # [+100, +50] -> 150, [-200, -300] -> -500, [+75] -> 75
        assert len(bursts) == 3
        assert bursts[0] == 150   # (100*1 + 50*1)
        assert bursts[1] == -500  # (200*-1 + 300*-1)
        assert bursts[2] == 75    # (75*1)

    def test_bursts_single_direction(self):
        """Test burst extraction with all same direction"""
        trace = [(0.0, 100, 1), (0.1, 200, 1), (0.2, 300, 1)]
        bursts = FeatureExtractor.extract_bursts(trace)
        assert len(bursts) == 1
        assert bursts[0] == 600

    def test_bayes_features_histogram(self):
        """Test Bayes histogram features"""
        features = FeatureExtractor.extract_bayes_features(self.simple_trace, interval=100)
        # Should be a list of ints, histogram over [-1500, 1501, 100]
        assert isinstance(features, list)
        assert len(features) > 0
        # Total should equal number of packets
        assert sum(features) == 3

    def test_vng_features(self):
        """Test VNG++ features include statistics + histogram"""
        features = FeatureExtractor.extract_vng_features(self.burst_trace, interval=5000)
        assert isinstance(features, list)
        # First 3 elements are [trace_time, up_bytes, down_bytes]
        assert features[1] == 225   # up: 100+50+75
        assert features[2] == 500   # down: 200+300
        # Trace time
        assert abs(features[0] - 0.4) < 0.01

    def test_svm_features(self):
        """Test SVM features include stats + burst histogram"""
        features = FeatureExtractor.extract_svm_features(self.simple_trace, interval=5000)
        assert isinstance(features, list)
        # First 5: [up_bytes, down_bytes, in_ratio, pack_num, burst_num]
        assert features[0] == 230   # up: 150+80
        assert features[1] == 1200  # down: 1200
        assert abs(features[2] - 1/3) < 0.01  # in_ratio: 1 of 3 packets

    def test_empty_trace(self):
        """Test handling of empty trace"""
        empty_trace = []

        ll_features = FeatureExtractor.extract_ll_features(empty_trace)
        assert len(ll_features) == 0

        bursts = FeatureExtractor.extract_bursts(empty_trace)
        assert len(bursts) == 0

    def test_single_packet_trace(self):
        """Test with single packet"""
        trace = [(0.0, 100, 1)]

        ll_features = FeatureExtractor.extract_ll_features(trace)
        assert len(ll_features) == 1
        assert 100 in ll_features

        bursts = FeatureExtractor.extract_bursts(trace)
        assert len(bursts) == 1
        assert bursts[0] == 100


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, '-v'])
