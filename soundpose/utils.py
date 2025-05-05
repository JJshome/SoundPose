"""
Utility functions for SoundPose.

This module contains utility functions for audio processing, visualization,
and other helper functions used in the SoundPose framework.
"""

import os
import logging
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def preprocess_audio(
    file_path: str,
    target_sr: int = 22050,
    segment_length: float = 5.0,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Preprocess audio file for analysis.
    
    Args:
        file_path: Path to the audio file
        target_sr: Target sample rate
        segment_length: Length of each audio segment in seconds
        normalize: Whether to normalize the audio
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    logger.info(f"Preprocessing audio file: {file_path}")
    
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=target_sr)
        
        # Check if audio is too short
        if len(audio) < int(target_sr * segment_length):
            # Pad with zeros if audio is shorter than segment_length
            padding = int(target_sr * segment_length) - len(audio)
            audio = np.pad(audio, (0, padding), mode="constant")
            logger.info(f"Audio was shorter than {segment_length}s, padded with zeros")
        
        # Normalize if requested
        if normalize:
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
        logger.info(f"Preprocessing complete. Audio length: {len(audio)/target_sr:.2f}s, Sample rate: {target_sr}")
        
        return audio, target_sr
    
    except Exception as e:
        logger.error(f"Error preprocessing audio file: {e}")
        raise


def compute_spectrogram(
    audio: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: int = 20,
    fmax: int = 8000,
    power: float = 2.0,
    db_scale: bool = True,
) -> np.ndarray:
    """
    Compute mel spectrogram from audio data.
    
    Args:
        audio: Audio data as numpy array
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency
        power: Power for spectrogram
        db_scale: Whether to convert to decibel scale
        
    Returns:
        Mel spectrogram as numpy array of shape (n_mels, time)
    """
    logger.info("Computing spectrogram")
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=power,
    )
    
    # Convert to decibel scale if requested
    if db_scale:
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    logger.info(f"Spectrogram computed with shape: {mel_spec.shape}")
    
    return mel_spec


def plot_spectrogram(
    spectrogram: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    fmin: int = 20,
    fmax: int = 8000,
    figsize: Tuple[int, int] = (10, 4),
    cmap: str = "magma",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot mel spectrogram.
    
    Args:
        spectrogram: Mel spectrogram as numpy array
        sr: Sample rate
        hop_length: Hop length used for spectrogram
        fmin: Minimum frequency
        fmax: Maximum frequency
        figsize: Figure size
        cmap: Colormap
        title: Plot title
        save_path: Path to save the plot (if None, plot is displayed)
    """
    plt.figure(figsize=figsize)
    
    librosa.display.specshow(
        spectrogram,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        fmin=fmin,
        fmax=fmax,
        cmap=cmap,
    )
    
    plt.colorbar(format="%+2.0f dB")
    
    if title:
        plt.title(title)
    else:
        plt.title("Mel Spectrogram")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Spectrogram saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_anomaly_scores(
    anomaly_scores: np.ndarray,
    threshold: float = 0.5,
    figsize: Tuple[int, int] = (10, 4),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot anomaly scores.
    
    Args:
        anomaly_scores: Anomaly scores as numpy array
        threshold: Threshold for anomaly detection
        figsize: Figure size
        title: Plot title
        save_path: Path to save the plot (if None, plot is displayed)
    """
    plt.figure(figsize=figsize)
    
    time_axis = np.arange(len(anomaly_scores))
    
    # Plot anomaly scores
    plt.plot(time_axis, anomaly_scores, color="blue", label="Anomaly Score")
    
    # Plot threshold
    plt.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold ({threshold:.2f})")
    
    # Highlight anomalies
    anomaly_indices = np.where(anomaly_scores > threshold)[0]
    if len(anomaly_indices) > 0:
        plt.scatter(
            anomaly_indices,
            anomaly_scores[anomaly_indices],
            color="red",
            alpha=0.7,
            s=50,
            label="Anomalies",
        )
    
    plt.xlabel("Time Frame")
    plt.ylabel("Anomaly Score")
    
    if title:
        plt.title(title)
    else:
        plt.title("Anomaly Scores")
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Anomaly scores plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def combine_plots(
    spectrogram: np.ndarray,
    anomaly_scores: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    fmin: int = 20,
    fmax: int = 8000,
    threshold: float = 0.5,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "magma",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Create a combined plot with spectrogram and anomaly scores.
    
    Args:
        spectrogram: Mel spectrogram as numpy array
        anomaly_scores: Anomaly scores as numpy array
        sr: Sample rate
        hop_length: Hop length used for spectrogram
        fmin: Minimum frequency
        fmax: Maximum frequency
        threshold: Threshold for anomaly detection
        figsize: Figure size
        cmap: Colormap for spectrogram
        title: Plot title
        save_path: Path to save the plot (if None, plot is displayed)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]})
    
    # Plot spectrogram
    img = librosa.display.specshow(
        spectrogram,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        fmin=fmin,
        fmax=fmax,
        cmap=cmap,
        ax=ax1,
    )
    
    fig.colorbar(img, ax=ax1, format="%+2.0f dB")
    ax1.set_title("Mel Spectrogram")
    
    # Plot anomaly scores
    time_axis = np.arange(len(anomaly_scores))
    ax2.plot(time_axis, anomaly_scores, color="blue", label="Anomaly Score")
    
    # Plot threshold
    ax2.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold ({threshold:.2f})")
    
    # Highlight anomalies
    anomaly_indices = np.where(anomaly_scores > threshold)[0]
    if len(anomaly_indices) > 0:
        ax2.scatter(
            anomaly_indices,
            anomaly_scores[anomaly_indices],
            color="red",
            alpha=0.7,
            s=50,
            label="Anomalies",
        )
    
    ax2.set_xlabel("Time Frame")
    ax2.set_ylabel("Anomaly Score")
    ax2.set_title("Anomaly Scores")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Combined plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def segment_audio(
    audio: np.ndarray,
    sr: int,
    segment_length: float = 5.0,
    overlap: float = 2.5,
) -> List[np.ndarray]:
    """
    Segment audio into overlapping chunks.
    
    Args:
        audio: Audio data as numpy array
        sr: Sample rate
        segment_length: Length of each segment in seconds
        overlap: Overlap between segments in seconds
        
    Returns:
        List of audio segments
    """
    segment_samples = int(segment_length * sr)
    overlap_samples = int(overlap * sr)
    hop_samples = segment_samples - overlap_samples
    
    # Calculate the number of segments
    num_segments = max(1, 1 + (len(audio) - segment_samples) // hop_samples)
    
    segments = []
    
    for i in range(num_segments):
        start = i * hop_samples
        end = start + segment_samples
        
        # Handle the last segment
        if end > len(audio):
            # Pad with zeros if needed
            segment = np.pad(audio[start:], (0, end - len(audio)), mode="constant")
        else:
            segment = audio[start:end]
        
        segments.append(segment)
    
    logger.info(f"Audio segmented into {len(segments)} segments of {segment_length}s with {overlap}s overlap")
    
    return segments


def normalize_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    """
    Normalize spectrogram to have zero mean and unit variance.
    
    Args:
        spectrogram: Mel spectrogram as numpy array
        
    Returns:
        Normalized spectrogram
    """
    mean = np.mean(spectrogram)
    std = np.std(spectrogram)
    return (spectrogram - mean) / (std + 1e-8)


def augment_spectrogram(spectrogram: np.ndarray, augmentation_type: str = "all") -> np.ndarray:
    """
    Apply augmentation to spectrogram.
    
    Args:
        spectrogram: Mel spectrogram as numpy array
        augmentation_type: Type of augmentation to apply
            ('time_mask', 'freq_mask', 'time_warp', 'all')
        
    Returns:
        Augmented spectrogram
    """
    augmented = spectrogram.copy()
    
    if augmentation_type in ["time_mask", "all"]:
        # Time masking
        time_mask_width = int(augmented.shape[1] * 0.1)
        num_masks = np.random.randint(1, 3)
        
        for _ in range(num_masks):
            start = np.random.randint(0, augmented.shape[1] - time_mask_width)
            augmented[:, start:start + time_mask_width] = np.min(augmented)
    
    if augmentation_type in ["freq_mask", "all"]:
        # Frequency masking
        freq_mask_width = int(augmented.shape[0] * 0.1)
        num_masks = np.random.randint(1, 3)
        
        for _ in range(num_masks):
            start = np.random.randint(0, augmented.shape[0] - freq_mask_width)
            augmented[start:start + freq_mask_width, :] = np.min(augmented)
    
    return augmented


def compute_feature_statistics(features: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute statistics of features for analysis.
    
    Args:
        features: Features as numpy array of shape (batch_size, seq_len, feature_dim)
        
    Returns:
        Dictionary of statistics (mean, std, min, max)
    """
    return {
        "mean": np.mean(features, axis=1),
        "std": np.std(features, axis=1),
        "min": np.min(features, axis=1),
        "max": np.max(features, axis=1),
    }


def save_audio(audio: np.ndarray, sr: int, file_path: str) -> None:
    """
    Save audio to a file.
    
    Args:
        audio: Audio data as numpy array
        sr: Sample rate
        file_path: Path to save the audio file
    """
    import soundfile as sf
    
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    sf.write(file_path, audio, sr)
    logger.info(f"Audio saved to {file_path}")
