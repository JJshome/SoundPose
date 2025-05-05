"""
Real-time monitoring example for SoundPose.

This example demonstrates how to use SoundPose for real-time monitoring
of audio streams and detection of anomalies.
"""

import os
import sys
import argparse
import logging
import time
import datetime

# Add parent directory to the path to import SoundPose
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from soundpose import SoundPoseMonitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def results_callback(results):
    """
    Callback function for processing analysis results.
    
    Args:
        results: AnalysisResults object
    """
    # If anomalies are detected, print information
    if results.has_anomalies():
        severity = results.get_severity_level()
        score = results.get_mean_anomaly_score()
        
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
              f"Anomaly detected! Severity: {severity}, Score: {score:.4f}")


def main():
    """Run the real-time monitoring example."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SoundPose Real-time Monitoring Example")
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.75,
        help="Threshold for anomaly detection"
    )
    parser.add_argument(
        "--window", "-w", type=int, default=2000,
        help="Analysis window size in milliseconds"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="voice", choices=["voice", "mechanical"],
        help="Type of model to use"
    )
    parser.add_argument(
        "--device", "-d", type=int, default=None,
        help="Audio input device index (default: system default)"
    )
    parser.add_argument(
        "--sample-rate", "-sr", type=int, default=22050,
        help="Sample rate to use"
    )
    parser.add_argument(
        "--duration", type=int, default=60,
        help="Monitoring duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--report", "-r", type=str, default=None,
        help="Path to save the monitoring report (if None, report is not saved)"
    )
    parser.add_argument(
        "--plot", "-p", type=str, default=None,
        help="Path to save the anomaly scores plot (if None, plot is not saved)"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting real-time monitoring")
    logger.info(f"Using model: {args.model}")
    logger.info(f"Using threshold: {args.threshold}")
    logger.info(f"Using window size: {args.window}ms")
    logger.info(f"Monitoring duration: {args.duration}s")
    
    # Create monitor
    monitor = SoundPoseMonitor(
        threshold=args.threshold,
        window_size=args.window,
        model_type=args.model,
        device=args.device,
        sample_rate=args.sample_rate,
        results_callback=results_callback,
    )
    
    try:
        # Start monitoring
        monitor.start()
        
        print("\nMonitoring started. Press Ctrl+C to stop.\n")
        
        # Monitor for the specified duration
        start_time = time.time()
        while time.time() - start_time < args.duration:
            # Check if any anomalies have been detected
            anomalies = monitor.get_detected_anomalies()
            
            # Sleep to avoid high CPU usage
            time.sleep(0.1)
        
        # Stop monitoring
        monitor.stop()
        
        # Generate report
        report = monitor.generate_report()
        
        # Print summary
        print("\n" + "="*50)
        print(f"Monitoring complete. Duration: {report['duration_seconds']:.2f}s")
        print(f"Number of samples: {report['num_samples']}")
        print(f"Number of anomalies: {report['num_anomalies']}")
        print(f"Anomaly rate: {report['anomaly_rate']*100:.2f}%")
        print(f"Mean anomaly score: {report['mean_anomaly_score']:.4f}")
        print(f"Max anomaly score: {report['max_anomaly_score']:.4f}")
        print("="*50 + "\n")
        
        # Save report if requested
        if args.report:
            directory = os.path.dirname(args.report)
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            monitor.save_report(args.report)
            logger.info(f"Monitoring report saved to {args.report}")
        
        # Plot anomaly scores if requested
        if args.plot:
            directory = os.path.dirname(args.plot)
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            monitor.plot_anomaly_scores(
                title="Real-time Monitoring Anomaly Scores",
                save_path=args.plot,
            )
            logger.info(f"Anomaly scores plot saved to {args.plot}")
        
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    
    finally:
        # Ensure monitoring is stopped
        if monitor.is_active():
            monitor.stop()


if __name__ == "__main__":
    main()
