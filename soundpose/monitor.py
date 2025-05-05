"""
Monitor module for SoundPose.

This module contains the SoundPoseMonitor class, which provides real-time
monitoring capabilities for audio streams.
"""

import os
import logging
import numpy as np
import threading
import time
import queue
import json
from typing import Dict, List, Optional, Tuple, Union, Callable
import datetime
import sounddevice as sd

from soundpose.analyzer import SoundPoseAnalyzer
from soundpose.utils import compute_spectrogram

logger = logging.getLogger(__name__)


class SoundPoseMonitor:
    """
    Class for real-time monitoring of audio streams.
    
    This class provides methods for monitoring audio streams in real-time
    and detecting anomalies using the SoundPose framework.
    
    Attributes:
        threshold (float): Threshold for anomaly detection
        window_size (int): Size of the analysis window in milliseconds
        analyzer (SoundPoseAnalyzer): Analyzer for processing audio
        is_monitoring (bool): Whether monitoring is active
        monitoring_thread (Optional[threading.Thread]): Thread for monitoring
        audio_queue (queue.Queue): Queue for audio data
        results_callback (Optional[Callable]): Callback for results
    """
    
    def __init__(
        self,
        threshold: float = 0.75,
        window_size: int = 2000,
        model_type: str = "voice",
        device: Optional[Union[str, int]] = None,
        sample_rate: int = 22050,
        channels: int = 1,
        results_callback: Optional[Callable] = None,
    ):
        """
        Initialize the SoundPoseMonitor.
        
        Args:
            threshold: Threshold for anomaly detection
            window_size: Size of the analysis window in milliseconds
            model_type: Type of model to use ('voice', 'mechanical', etc.)
            device: Audio device to use (None for default)
            sample_rate: Sample rate to use
            channels: Number of audio channels
            results_callback: Callback function for results
        """
        self.threshold = threshold
        self.window_size = window_size
        self.model_type = model_type
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.results_callback = results_callback
        
        # Initialize analyzer
        self.analyzer = SoundPoseAnalyzer(model_type=model_type, threshold=threshold)
        
        # Initialize monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.audio_queue = queue.Queue()
        
        # Initialize results storage
        self.anomaly_scores = []
        self.timestamps = []
        self.detected_anomalies = []
        
        logger.info(f"SoundPoseMonitor initialized with window size: {window_size}ms, threshold: {threshold}")
    
    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Dict,
        status: sd.CallbackFlags,
    ) -> None:
        """
        Callback function for audio input.
        
        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Time information
            status: Status flags
        """
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Add audio data to queue
        self.audio_queue.put((indata.copy(), time_info))
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop that processes audio data."""
        logger.info("Monitoring loop started")
        
        # Calculate window parameters
        window_samples = int(self.window_size * self.sample_rate / 1000)
        hop_samples = window_samples // 2  # 50% overlap
        
        # Buffer for audio data
        audio_buffer = np.zeros(window_samples, dtype=np.float32)
        
        try:
            while self.is_monitoring:
                try:
                    # Get audio data from queue with timeout
                    indata, time_info = self.audio_queue.get(timeout=0.1)
                    
                    # Process audio data (if mono, squeeze the channel dimension)
                    if self.channels == 1:
                        audio_data = indata.squeeze()
                    else:
                        # Use first channel if multiple channels
                        audio_data = indata[:, 0]
                    
                    # Shift buffer and add new data
                    audio_buffer = np.roll(audio_buffer, -len(audio_data))
                    audio_buffer[-len(audio_data):] = audio_data
                    
                    # Compute spectrogram
                    spectrogram = compute_spectrogram(
                        audio=audio_buffer,
                        sr=self.sample_rate,
                        n_fft=1024,
                        hop_length=256,
                        n_mels=128,
                    )
                    
                    # Analyze with SoundPose
                    results = self.analyzer.analyze_audio(audio_buffer, self.sample_rate)
                    
                    # Get current timestamp
                    timestamp = datetime.datetime.now()
                    
                    # Store results
                    self.anomaly_scores.append(results.get_mean_anomaly_score())
                    self.timestamps.append(timestamp)
                    
                    # Check for anomalies
                    if results.has_anomalies():
                        logger.info(f"Anomaly detected at {timestamp}, score: {results.get_mean_anomaly_score():.4f}")
                        
                        # Add to detected anomalies
                        self.detected_anomalies.append({
                            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                            "score": results.get_mean_anomaly_score(),
                            "max_score": results.get_max_anomaly_score(),
                            "severity": results.get_severity_level(),
                        })
                        
                        # If too many anomalies, keep only the last 100
                        if len(self.detected_anomalies) > 100:
                            self.detected_anomalies = self.detected_anomalies[-100:]
                    
                    # Call results callback if provided
                    if self.results_callback:
                        self.results_callback(results)
                
                except queue.Empty:
                    # No audio data in queue, continue
                    pass
        
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            self.is_monitoring = False
        
        logger.info("Monitoring loop stopped")
    
    def start(self) -> None:
        """Start monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring is already active")
            return
        
        logger.info("Starting monitoring")
        
        # Clear previous results
        self.anomaly_scores = []
        self.timestamps = []
        self.detected_anomalies = []
        
        # Set monitoring flag
        self.is_monitoring = True
        
        # Start audio stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._audio_callback,
            blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
            device=self.device,
        )
        self.stream.start()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Monitoring started")
    
    def stop(self) -> None:
        """Stop monitoring."""
        if not self.is_monitoring:
            logger.warning("Monitoring is not active")
            return
        
        logger.info("Stopping monitoring")
        
        # Clear monitoring flag
        self.is_monitoring = False
        
        # Stop audio stream
        if hasattr(self, "stream"):
            self.stream.stop()
            self.stream.close()
        
        # Wait for monitoring thread to stop
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
        
        logger.info("Monitoring stopped")
    
    def is_active(self) -> bool:
        """
        Check if monitoring is active.
        
        Returns:
            True if monitoring is active, False otherwise
        """
        return self.is_monitoring
    
    def get_anomaly_scores(self) -> Tuple[List[datetime.datetime], List[float]]:
        """
        Get the anomaly scores with timestamps.
        
        Returns:
            Tuple of (timestamps, anomaly_scores)
        """
        return self.timestamps, self.anomaly_scores
    
    def get_detected_anomalies(self) -> List[Dict]:
        """
        Get the detected anomalies.
        
        Returns:
            List of detected anomalies
        """
        return self.detected_anomalies
    
    def generate_report(self) -> Dict:
        """
        Generate a report of the monitoring session.
        
        Returns:
            Dictionary with report information
        """
        end_time = datetime.datetime.now()
        
        # Calculate monitoring duration
        if self.timestamps:
            start_time = self.timestamps[0]
            duration = (end_time - start_time).total_seconds()
        else:
            start_time = end_time
            duration = 0
        
        # Calculate statistics
        if self.anomaly_scores:
            mean_score = np.mean(self.anomaly_scores)
            max_score = np.max(self.anomaly_scores)
            min_score = np.min(self.anomaly_scores)
            std_score = np.std(self.anomaly_scores)
        else:
            mean_score = 0.0
            max_score = 0.0
            min_score = 0.0
            std_score = 0.0
        
        return {
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration,
            "model_type": self.model_type,
            "threshold": self.threshold,
            "window_size_ms": self.window_size,
            "sample_rate": self.sample_rate,
            "num_samples": len(self.anomaly_scores),
            "num_anomalies": len(self.detected_anomalies),
            "anomaly_rate": len(self.detected_anomalies) / max(1, len(self.anomaly_scores)),
            "mean_anomaly_score": mean_score,
            "max_anomaly_score": max_score,
            "min_anomaly_score": min_score,
            "std_anomaly_score": std_score,
            "detected_anomalies": self.detected_anomalies,
        }
    
    def save_report(self, file_path: str) -> None:
        """
        Save the monitoring report to a file.
        
        Args:
            file_path: Path to save the report
        """
        report = self.generate_report()
        
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        with open(file_path, "w") as f:
            json.dump(report, f, indent=4)
        
        logger.info(f"Monitoring report saved to {file_path}")
    
    def plot_anomaly_scores(
        self,
        figsize: Tuple[int, int] = (10, 4),
        title: str = "Real-time Anomaly Scores",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot the anomaly scores from the monitoring session.
        
        Args:
            figsize: Figure size
            title: Plot title
            save_path: Path to save the plot (if None, plot is displayed)
        """
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter
        
        if not self.timestamps:
            logger.warning("No data to plot")
            return
        
        plt.figure(figsize=figsize)
        
        # Convert timestamps to matplotlib format
        times = [t for t in self.timestamps]
        
        # Plot anomaly scores
        plt.plot(times, self.anomaly_scores, label="Anomaly Score")
        
        # Plot threshold
        plt.axhline(
            y=self.threshold, color="red", linestyle="--", label=f"Threshold ({self.threshold:.2f})"
        )
        
        # Format x-axis
        plt.gcf().autofmt_xdate()
        date_format = DateFormatter("%H:%M:%S")
        plt.gca().xaxis.set_major_formatter(date_format)
        
        plt.xlabel("Time")
        plt.ylabel("Anomaly Score")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            directory = os.path.dirname(save_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Anomaly scores plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
