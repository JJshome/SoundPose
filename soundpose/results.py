"""
Results module for SoundPose.

This module contains the AnalysisResults class, which encapsulates
the results of audio analysis and provides methods for visualization
and interpretation.
"""

import os
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import json
import datetime

from soundpose.utils import plot_spectrogram, plot_anomaly_scores, combine_plots

logger = logging.getLogger(__name__)


class AnalysisResults:
    """
    Class for encapsulating and interpreting analysis results.
    
    This class provides methods for visualizing and interpreting the results
    of audio analysis, including spectrograms and anomaly scores.
    
    Attributes:
        file_path (Optional[str]): Path to the analyzed audio file
        features (torch.Tensor): Features extracted from the audio
        anomaly_scores (np.ndarray): Anomaly scores for each time frame
        spectrogram (np.ndarray): Mel spectrogram of the audio
        threshold (float): Threshold for anomaly detection
        sample_rate (int): Sample rate of the audio
        model_type (str): Type of model used for analysis
        timestamp (str): Timestamp of the analysis
    """
    
    def __init__(
        self,
        file_path: Optional[str],
        features: torch.Tensor,
        anomaly_scores: np.ndarray,
        spectrogram: np.ndarray,
        threshold: float,
        sample_rate: int,
        model_type: str,
    ):
        """
        Initialize the AnalysisResults.
        
        Args:
            file_path: Path to the analyzed audio file
            features: Features extracted from the audio
            anomaly_scores: Anomaly scores for each time frame
            spectrogram: Mel spectrogram of the audio
            threshold: Threshold for anomaly detection
            sample_rate: Sample rate of the audio
            model_type: Type of model used for analysis
        """
        self.file_path = file_path
        self.features = features.detach().cpu() if isinstance(features, torch.Tensor) else features
        self.anomaly_scores = anomaly_scores
        self.spectrogram = spectrogram
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.model_type = model_type
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Compute anomaly segments
        self.anomaly_segments = self._compute_anomaly_segments()
        
        logger.info("AnalysisResults initialized")
    
    def _compute_anomaly_segments(self) -> List[Dict[str, Union[int, float]]]:
        """
        Compute segments of continuous anomalies.
        
        Returns:
            List of dictionaries with start, end, and score of anomaly segments
        """
        segments = []
        is_anomaly = False
        start_idx = 0
        scores = []
        
        for i, score in enumerate(self.anomaly_scores):
            if score > self.threshold and not is_anomaly:
                # Start of anomaly
                is_anomaly = True
                start_idx = i
                scores = [score]
            elif score > self.threshold and is_anomaly:
                # Continuing anomaly
                scores.append(score)
            elif score <= self.threshold and is_anomaly:
                # End of anomaly
                is_anomaly = False
                
                # Only add if the segment is at least 3 frames
                if i - start_idx >= 3:
                    segments.append({
                        "start": start_idx,
                        "end": i - 1,
                        "score": np.mean(scores),
                        "max_score": np.max(scores),
                    })
                
                scores = []
        
        # Handle anomaly at the end
        if is_anomaly and len(scores) >= 3:
            segments.append({
                "start": start_idx,
                "end": len(self.anomaly_scores) - 1,
                "score": np.mean(scores),
                "max_score": np.max(scores),
            })
        
        return segments
    
    def get_mean_anomaly_score(self) -> float:
        """
        Get the mean anomaly score.
        
        Returns:
            Mean anomaly score
        """
        return float(np.mean(self.anomaly_scores))
    
    def get_max_anomaly_score(self) -> float:
        """
        Get the maximum anomaly score.
        
        Returns:
            Maximum anomaly score
        """
        return float(np.max(self.anomaly_scores))
    
    def has_anomalies(self) -> bool:
        """
        Check if any anomalies were detected.
        
        Returns:
            True if anomalies were detected, False otherwise
        """
        return len(self.anomaly_segments) > 0
    
    def get_anomaly_segments(self) -> List[Dict[str, Union[int, float]]]:
        """
        Get the list of anomaly segments.
        
        Returns:
            List of dictionaries with start, end, and score of anomaly segments
        """
        return self.anomaly_segments
    
    def get_anomaly_percentage(self) -> float:
        """
        Get the percentage of time frames with anomalies.
        
        Returns:
            Percentage of time frames with anomalies
        """
        if len(self.anomaly_scores) == 0:
            return 0.0
        
        anomaly_frames = np.sum(self.anomaly_scores > self.threshold)
        return float(anomaly_frames / len(self.anomaly_scores) * 100)
    
    def plot_spectrogram(
        self,
        figsize: Tuple[int, int] = (10, 4),
        cmap: str = "magma",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot the spectrogram.
        
        Args:
            figsize: Figure size
            cmap: Colormap
            title: Plot title
            save_path: Path to save the plot (if None, plot is displayed)
        """
        file_name = os.path.basename(self.file_path) if self.file_path else "unknown"
        
        if title is None:
            title = f"Spectrogram - {file_name}"
        
        plot_spectrogram(
            spectrogram=self.spectrogram,
            sr=self.sample_rate,
            hop_length=512,  # Default hop length
            figsize=figsize,
            cmap=cmap,
            title=title,
            save_path=save_path,
        )
    
    def plot_anomaly_scores(
        self,
        figsize: Tuple[int, int] = (10, 4),
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot the anomaly scores.
        
        Args:
            figsize: Figure size
            title: Plot title
            save_path: Path to save the plot (if None, plot is displayed)
        """
        file_name = os.path.basename(self.file_path) if self.file_path else "unknown"
        
        if title is None:
            title = f"Anomaly Scores - {file_name}"
        
        plot_anomaly_scores(
            anomaly_scores=self.anomaly_scores,
            threshold=self.threshold,
            figsize=figsize,
            title=title,
            save_path=save_path,
        )
    
    def plot_combined(
        self,
        figsize: Tuple[int, int] = (12, 8),
        cmap: str = "magma",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create a combined plot with spectrogram and anomaly scores.
        
        Args:
            figsize: Figure size
            cmap: Colormap for spectrogram
            title: Plot title
            save_path: Path to save the plot (if None, plot is displayed)
        """
        file_name = os.path.basename(self.file_path) if self.file_path else "unknown"
        
        if title is None:
            title = f"Analysis Results - {file_name}"
        
        combine_plots(
            spectrogram=self.spectrogram,
            anomaly_scores=self.anomaly_scores,
            sr=self.sample_rate,
            hop_length=512,  # Default hop length
            threshold=self.threshold,
            figsize=figsize,
            cmap=cmap,
            title=title,
            save_path=save_path,
        )
    
    def get_summary(self) -> Dict[str, Union[str, float, int, List[Dict[str, Union[int, float]]]]]:
        """
        Get a summary of the analysis results.
        
        Returns:
            Dictionary with summary information
        """
        file_name = os.path.basename(self.file_path) if self.file_path else "unknown"
        
        # Convert time frames to seconds
        hop_length = 512  # Default hop length
        frame_duration = hop_length / self.sample_rate
        
        # Convert anomaly segments to time in seconds
        time_segments = []
        for segment in self.anomaly_segments:
            time_segments.append({
                "start_time": segment["start"] * frame_duration,
                "end_time": segment["end"] * frame_duration,
                "duration": (segment["end"] - segment["start"]) * frame_duration,
                "score": segment["score"],
                "max_score": segment["max_score"],
            })
        
        return {
            "file_name": file_name,
            "file_path": self.file_path,
            "timestamp": self.timestamp,
            "model_type": self.model_type,
            "sample_rate": self.sample_rate,
            "mean_anomaly_score": self.get_mean_anomaly_score(),
            "max_anomaly_score": self.get_max_anomaly_score(),
            "has_anomalies": self.has_anomalies(),
            "anomaly_percentage": self.get_anomaly_percentage(),
            "anomaly_segments": time_segments,
            "threshold": self.threshold,
        }
    
    def save_summary(self, file_path: str) -> None:
        """
        Save the summary to a JSON file.
        
        Args:
            file_path: Path to save the summary
        """
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        with open(file_path, "w") as f:
            json.dump(self.get_summary(), f, indent=4)
        
        logger.info(f"Summary saved to {file_path}")
    
    def save_results(self, directory: str) -> None:
        """
        Save the analysis results to a directory.
        
        This method saves the spectrogram, anomaly scores plot, combined plot,
        and summary to the specified directory.
        
        Args:
            directory: Directory to save the results
        """
        os.makedirs(directory, exist_ok=True)
        
        file_name = os.path.basename(self.file_path) if self.file_path else "unknown"
        base_name = os.path.splitext(file_name)[0]
        
        # Save plots
        self.plot_spectrogram(save_path=os.path.join(directory, f"{base_name}_spectrogram.png"))
        self.plot_anomaly_scores(save_path=os.path.join(directory, f"{base_name}_anomaly_scores.png"))
        self.plot_combined(save_path=os.path.join(directory, f"{base_name}_combined.png"))
        
        # Save summary
        self.save_summary(os.path.join(directory, f"{base_name}_summary.json"))
        
        logger.info(f"Results saved to {directory}")
    
    def get_severity_level(self) -> str:
        """
        Get the severity level based on anomaly scores.
        
        Returns:
            Severity level (None, Low, Medium, High, Severe)
        """
        if not self.has_anomalies():
            return "None"
        
        max_score = self.get_max_anomaly_score()
        
        if max_score < self.threshold * 1.2:
            return "Low"
        elif max_score < self.threshold * 1.5:
            return "Medium"
        elif max_score < self.threshold * 2.0:
            return "High"
        else:
            return "Severe"
    
    def get_diagnostic_report(self) -> str:
        """
        Generate a diagnostic report based on the analysis results.
        
        Returns:
            Diagnostic report as a formatted string
        """
        file_name = os.path.basename(self.file_path) if self.file_path else "unknown"
        severity = self.get_severity_level()
        anomaly_percentage = self.get_anomaly_percentage()
        
        report = [
            f"Diagnostic Report for {file_name}",
            f"Timestamp: {self.timestamp}",
            f"Model Type: {self.model_type}",
            "",
            f"Severity Level: {severity}",
            f"Mean Anomaly Score: {self.get_mean_anomaly_score():.4f}",
            f"Max Anomaly Score: {self.get_max_anomaly_score():.4f}",
            f"Anomaly Percentage: {anomaly_percentage:.2f}%",
            "",
        ]
        
        if self.has_anomalies():
            report.append(f"Detected {len(self.anomaly_segments)} anomaly segments:")
            
            # Convert time frames to seconds
            hop_length = 512  # Default hop length
            frame_duration = hop_length / self.sample_rate
            
            for i, segment in enumerate(self.anomaly_segments):
                start_time = segment["start"] * frame_duration
                end_time = segment["end"] * frame_duration
                duration = end_time - start_time
                
                report.append(
                    f"  Segment {i+1}: {start_time:.2f}s - {end_time:.2f}s "
                    f"(duration: {duration:.2f}s, score: {segment['score']:.4f})"
                )
        else:
            report.append("No anomalies detected.")
        
        if self.model_type == "voice" and severity != "None":
            report.append("")
            report.append("Possible voice-related conditions to investigate:")
            
            if severity == "Low":
                report.append("  - Minor vocal strain or fatigue")
                report.append("  - Early signs of vocal fold irritation")
            elif severity == "Medium":
                report.append("  - Vocal fold inflammation or swelling")
                report.append("  - Mild muscle tension dysphonia")
            elif severity == "High":
                report.append("  - Possible vocal fold nodules or polyps")
                report.append("  - Significant muscle tension dysphonia")
            else:  # Severe
                report.append("  - Vocal fold paralysis or paresis")
                report.append("  - Vocal fold nodules, polyps, or cysts")
                report.append("  - Spasmodic dysphonia")
            
            report.append("")
            report.append("Note: This is an automated analysis and should not be used")
            report.append("for medical diagnosis. Please consult a healthcare professional.")
        
        return "\n".join(report)
