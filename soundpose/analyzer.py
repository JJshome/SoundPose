"""
Core analyzer module for SoundPose.

This module contains the main SoundPoseAnalyzer class, which is responsible for
analyzing audio files and detecting anomalies using transformer-based architecture.
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import librosa
import transformers

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from soundpose.models import TransformerEncoder, GenerativeModel
from soundpose.baseline import BaselineBuilder
from soundpose.utils import preprocess_audio, compute_spectrogram
from soundpose.results import AnalysisResults

logger = logging.getLogger(__name__)


class SoundPoseAnalyzer:
    """
    Main analyzer class for SoundPose.
    
    This class provides methods to analyze audio files and detect anomalies
    using transformer-based architecture and generative modeling.
    
    Attributes:
        model_type (str): Type of model to use ('voice', 'mechanical', etc.)
        baseline (Optional[BaselineBuilder]): Baseline for personalized analysis
        device (torch.device): Device to use for computation
        transformer (TransformerEncoder): Transformer model for feature extraction
        generative_model (GenerativeModel): Generative model for anomaly detection
        threshold (float): Threshold for anomaly detection
    """
    
    def __init__(
        self,
        model_type: str = "voice",
        baseline: Optional[BaselineBuilder] = None,
        threshold: float = 0.75,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize the SoundPoseAnalyzer.
        
        Args:
            model_type: Type of model to use ('voice', 'mechanical', etc.)
            baseline: Baseline for personalized analysis
            threshold: Threshold for anomaly detection
            device: Device to use for computation
        """
        self.model_type = model_type
        self.baseline = baseline
        self.threshold = threshold
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        logger.info(f"Initializing SoundPoseAnalyzer with model type: {model_type}")
        logger.info(f"Using device: {self.device}")
        
        # Load models
        self._load_models()
        
        # Set model parameters based on model type
        self._set_model_parameters()
        
        logger.info("SoundPoseAnalyzer initialized successfully")
        
    def _load_models(self):
        """Load transformer and generative models."""
        
        # Initialize the transformer encoder
        self.transformer = TransformerEncoder(
            model_type=self.model_type,
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            dropout=0.1,
        ).to(self.device)
        
        # Initialize the generative model for anomaly detection
        self.generative_model = GenerativeModel(
            model_type=self.model_type,
            latent_dim=256,
            device=self.device,
        ).to(self.device)
        
        # Load pre-trained weights if available
        model_dir = Path(__file__).parent / "pretrained_models"
        
        transformer_path = model_dir / f"transformer_{self.model_type}.pt"
        if transformer_path.exists():
            self.transformer.load_state_dict(torch.load(transformer_path, map_location=self.device))
            logger.info(f"Loaded transformer model from {transformer_path}")
        else:
            logger.warning(f"No pre-trained transformer model found at {transformer_path}. Using random initialization.")
        
        generative_path = model_dir / f"generative_{self.model_type}.pt"
        if generative_path.exists():
            self.generative_model.load_state_dict(torch.load(generative_path, map_location=self.device))
            logger.info(f"Loaded generative model from {generative_path}")
        else:
            logger.warning(f"No pre-trained generative model found at {generative_path}. Using random initialization.")
            
        # Set models to evaluation mode
        self.transformer.eval()
        self.generative_model.eval()
    
    def _set_model_parameters(self):
        """Set model parameters based on model type."""
        
        if self.model_type == "voice":
            self.sample_rate = 16000
            self.n_fft = 1024
            self.hop_length = 512
            self.n_mels = 128
            self.min_freq = 20
            self.max_freq = 8000
        elif self.model_type == "mechanical":
            self.sample_rate = 44100
            self.n_fft = 2048
            self.hop_length = 1024
            self.n_mels = 128
            self.min_freq = 20
            self.max_freq = 20000
        else:
            # Default parameters
            self.sample_rate = 22050
            self.n_fft = 1024
            self.hop_length = 512
            self.n_mels = 128
            self.min_freq = 20
            self.max_freq = 11025
    
    def analyze_file(self, file_path: str, segment_length: float = 5.0) -> AnalysisResults:
        """
        Analyze an audio file and detect anomalies.
        
        Args:
            file_path: Path to the audio file
            segment_length: Length of each audio segment in seconds
            
        Returns:
            AnalysisResults object containing the analysis results
        """
        logger.info(f"Analyzing file: {file_path}")
        
        # Load and preprocess audio
        audio, sr = preprocess_audio(
            file_path, 
            target_sr=self.sample_rate, 
            segment_length=segment_length
        )
        
        # Compute spectrogram
        spectrogram = compute_spectrogram(
            audio, 
            sr=self.sample_rate, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels,
            fmin=self.min_freq, 
            fmax=self.max_freq
        )
        
        # Extract features using transformer
        features = self._extract_features(spectrogram)
        
        # Detect anomalies using generative model
        anomaly_scores = self._detect_anomalies(features)
        
        # Create results object
        results = AnalysisResults(
            file_path=file_path,
            features=features,
            anomaly_scores=anomaly_scores,
            spectrogram=spectrogram,
            threshold=self.threshold,
            sample_rate=self.sample_rate,
            model_type=self.model_type
        )
        
        logger.info(f"Analysis complete. Mean anomaly score: {results.get_mean_anomaly_score():.4f}")
        
        return results
    
    def analyze_audio(self, audio: np.ndarray, sr: int) -> AnalysisResults:
        """
        Analyze raw audio data and detect anomalies.
        
        Args:
            audio: Audio data as numpy array
            sr: Sample rate of the audio
            
        Returns:
            AnalysisResults object containing the analysis results
        """
        logger.info("Analyzing raw audio data")
        
        # Resample if necessary
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        # Compute spectrogram
        spectrogram = compute_spectrogram(
            audio, 
            sr=sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels,
            fmin=self.min_freq, 
            fmax=self.max_freq
        )
        
        # Extract features using transformer
        features = self._extract_features(spectrogram)
        
        # Detect anomalies using generative model
        anomaly_scores = self._detect_anomalies(features)
        
        # Create results object
        results = AnalysisResults(
            file_path=None,
            features=features,
            anomaly_scores=anomaly_scores,
            spectrogram=spectrogram,
            threshold=self.threshold,
            sample_rate=sr,
            model_type=self.model_type
        )
        
        logger.info(f"Analysis complete. Mean anomaly score: {results.get_mean_anomaly_score():.4f}")
        
        return results
    
    def _extract_features(self, spectrogram: np.ndarray) -> torch.Tensor:
        """
        Extract features from spectrogram using transformer model.
        
        Args:
            spectrogram: Mel spectrogram of the audio
            
        Returns:
            Extracted features
        """
        # Convert to tensor and add batch dimension
        spec_tensor = torch.from_numpy(spectrogram).float().unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.transformer(spec_tensor)
            
        return features
    
    def _detect_anomalies(self, features: torch.Tensor) -> np.ndarray:
        """
        Detect anomalies in the extracted features.
        
        Args:
            features: Features extracted from the transformer
            
        Returns:
            Anomaly scores
        """
        with torch.no_grad():
            # Generate expected normal features
            generated_features = self.generative_model(features)
            
            # Compute residual loss (anomaly score)
            residual = torch.abs(features - generated_features)
            anomaly_scores = residual.mean(dim=2).squeeze(0).cpu().numpy()
            
        return anomaly_scores
    
    def set_threshold(self, threshold: float):
        """
        Set the threshold for anomaly detection.
        
        Args:
            threshold: New threshold value
        """
        self.threshold = threshold
        logger.info(f"Threshold set to {threshold}")
    
    def save_models(self, path: str):
        """
        Save the models to the specified path.
        
        Args:
            path: Directory path to save models
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        transformer_path = path / f"transformer_{self.model_type}.pt"
        generative_path = path / f"generative_{self.model_type}.pt"
        
        torch.save(self.transformer.state_dict(), transformer_path)
        torch.save(self.generative_model.state_dict(), generative_path)
        
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str):
        """
        Load models from the specified path.
        
        Args:
            path: Directory path containing the models
        """
        path = Path(path)
        
        transformer_path = path / f"transformer_{self.model_type}.pt"
        generative_path = path / f"generative_{self.model_type}.pt"
        
        if transformer_path.exists():
            self.transformer.load_state_dict(torch.load(transformer_path, map_location=self.device))
            logger.info(f"Loaded transformer model from {transformer_path}")
        else:
            raise FileNotFoundError(f"No transformer model found at {transformer_path}")
        
        if generative_path.exists():
            self.generative_model.load_state_dict(torch.load(generative_path, map_location=self.device))
            logger.info(f"Loaded generative model from {generative_path}")
        else:
            raise FileNotFoundError(f"No generative model found at {generative_path}")
        
        # Set models to evaluation mode
        self.transformer.eval()
        self.generative_model.eval()
