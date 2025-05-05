"""
Baseline module for SoundPose.

This module contains the BaselineBuilder class, which is used to create
personalized baselines for anomaly detection.
"""

import os
import logging
import numpy as np
import torch
import pickle
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from soundpose.utils import preprocess_audio, compute_spectrogram

logger = logging.getLogger(__name__)


class BaselineBuilder:
    """
    Class for building personalized baselines for anomaly detection.
    
    A baseline is a statistical model of normal audio patterns for a specific
    individual or source. It is used to detect deviations from the normal
    patterns, which may indicate anomalies.
    
    Attributes:
        recordings (List[Dict]): List of recordings added to the baseline
        features (Optional[np.ndarray]): Extracted features from recordings
        feature_stats (Optional[Dict]): Statistics of the features
        is_built (bool): Whether the baseline has been built
    """
    
    def __init__(self):
        """Initialize the BaselineBuilder."""
        self.recordings = []
        self.features = None
        self.feature_stats = None
        self.is_built = False
        
        logger.info("BaselineBuilder initialized")
    
    def add_recording(
        self,
        file_path: str,
        sample_rate: int = 22050,
        segment_length: float = 5.0,
    ) -> None:
        """
        Add a recording to the baseline.
        
        Args:
            file_path: Path to the audio file
            sample_rate: Target sample rate
            segment_length: Length of each audio segment in seconds
        """
        logger.info(f"Adding recording to baseline: {file_path}")
        
        # Load and preprocess audio
        audio, sr = preprocess_audio(
            file_path=file_path,
            target_sr=sample_rate,
            segment_length=segment_length,
        )
        
        # Compute spectrogram
        spectrogram = compute_spectrogram(
            audio=audio,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
        )
        
        # Add to recordings
        self.recordings.append({
            "file_path": file_path,
            "spectrogram": spectrogram,
            "sample_rate": sr,
        })
        
        # Reset built status
        self.is_built = False
        
        logger.info(f"Recording added. Total recordings: {len(self.recordings)}")
    
    def add_spectrograms(self, spectrograms: List[np.ndarray]) -> None:
        """
        Add pre-computed spectrograms to the baseline.
        
        Args:
            spectrograms: List of mel spectrograms as numpy arrays
        """
        logger.info(f"Adding {len(spectrograms)} pre-computed spectrograms to baseline")
        
        for i, spectrogram in enumerate(spectrograms):
            self.recordings.append({
                "file_path": f"spectrogram_{i}",
                "spectrogram": spectrogram,
                "sample_rate": None,
            })
        
        # Reset built status
        self.is_built = False
        
        logger.info(f"Spectrograms added. Total recordings: {len(self.recordings)}")
    
    def build(self, transformer=None) -> None:
        """
        Build the baseline using the added recordings.
        
        Args:
            transformer: Optional transformer model for feature extraction
        """
        if len(self.recordings) == 0:
            raise ValueError("No recordings added to the baseline")
        
        logger.info(f"Building baseline with {len(self.recordings)} recordings")
        
        if transformer is None:
            # Use spectrograms as features
            self.features = np.array([recording["spectrogram"] for recording in self.recordings])
            
            # Compute statistics
            self._compute_statistics()
        else:
            # Use transformer to extract features
            spectrograms = [recording["spectrogram"] for recording in self.recordings]
            features = []
            
            for spectrogram in spectrograms:
                # Convert to tensor and add batch dimension
                spec_tensor = torch.from_numpy(spectrogram).float().unsqueeze(0).to(transformer.device)
                
                # Extract features
                with torch.no_grad():
                    feature = transformer(spec_tensor).cpu().numpy()
                    features.append(feature.squeeze(0))
            
            self.features = np.array(features)
            
            # Compute statistics
            self._compute_statistics()
        
        self.is_built = True
        
        logger.info("Baseline built successfully")
    
    def _compute_statistics(self) -> None:
        """Compute statistics of the features."""
        logger.info("Computing feature statistics")
        
        self.feature_stats = {
            "mean": np.mean(self.features, axis=0),
            "std": np.std(self.features, axis=0),
            "min": np.min(self.features, axis=0),
            "max": np.max(self.features, axis=0),
        }
    
    def get_statistics(self) -> Dict[str, np.ndarray]:
        """
        Get the statistics of the baseline features.
        
        Returns:
            Dictionary of statistics
        """
        if not self.is_built:
            raise ValueError("Baseline has not been built yet")
        
        return self.feature_stats
    
    def save(self, file_path: str) -> None:
        """
        Save the baseline to a file.
        
        Args:
            file_path: Path to save the baseline
        """
        if not self.is_built:
            raise ValueError("Baseline has not been built yet")
        
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        with open(file_path, "wb") as f:
            pickle.dump({
                "features": self.features,
                "feature_stats": self.feature_stats,
                "is_built": self.is_built,
            }, f)
        
        logger.info(f"Baseline saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> "BaselineBuilder":
        """
        Load a baseline from a file.
        
        Args:
            file_path: Path to the baseline file
            
        Returns:
            Loaded BaselineBuilder instance
        """
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        baseline = cls()
        baseline.features = data["features"]
        baseline.feature_stats = data["feature_stats"]
        baseline.is_built = data["is_built"]
        
        logger.info(f"Baseline loaded from {file_path}")
        
        return baseline
    
    def compute_anomaly_scores(self, features: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores based on the baseline.
        
        Args:
            features: Features to compute anomaly scores for
            
        Returns:
            Anomaly scores
        """
        if not self.is_built:
            raise ValueError("Baseline has not been built yet")
        
        # Compute z-scores
        z_scores = (features - self.feature_stats["mean"]) / (self.feature_stats["std"] + 1e-8)
        
        # Compute anomaly scores (normalized Euclidean distance)
        anomaly_scores = np.sqrt(np.mean(z_scores ** 2, axis=-1))
        
        return anomaly_scores
