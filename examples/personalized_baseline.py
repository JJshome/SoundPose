"""
Personalized baseline example for SoundPose.

This example demonstrates how to build a personalized baseline for anomaly detection
and use it for analysis.
"""

import os
import sys
import argparse
import logging
import glob

# Add parent directory to the path to import SoundPose
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from soundpose import SoundPoseAnalyzer, BaselineBuilder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Run the personalized baseline example."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SoundPose Personalized Baseline Example")
    parser.add_argument(
        "--baseline-dir", "-b", type=str, required=True,
        help="Directory containing normal/baseline audio recordings"
    )
    parser.add_argument(
        "--test-file", "-t", type=str, required=True,
        help="Path to the audio file to analyze against the baseline"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.75,
        help="Threshold for anomaly detection"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="voice", choices=["voice", "mechanical"],
        help="Type of model to use"
    )
    parser.add_argument(
        "--save-baseline", "-s", type=str, default=None,
        help="Path to save the baseline (if None, baseline is not saved)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Directory to save the results (if None, results are displayed)"
    )
    
    args = parser.parse_args()
    
    # Check if the baseline directory exists
    if not os.path.isdir(args.baseline_dir):
        logger.error(f"Baseline directory not found: {args.baseline_dir}")
        return
    
    # Check if the test file exists
    if not os.path.isfile(args.test_file):
        logger.error(f"Test file not found: {args.test_file}")
        return
    
    logger.info(f"Building baseline from directory: {args.baseline_dir}")
    logger.info(f"Using model: {args.model}")
    logger.info(f"Using threshold: {args.threshold}")
    
    # Create baseline builder
    baseline = BaselineBuilder()
    
    # Find all audio files in the baseline directory
    audio_files = []
    for ext in [".wav", ".mp3", ".flac", ".ogg"]:
        audio_files.extend(glob.glob(os.path.join(args.baseline_dir, f"*{ext}")))
    
    if not audio_files:
        logger.error(f"No audio files found in baseline directory: {args.baseline_dir}")
        return
    
    logger.info(f"Found {len(audio_files)} audio files for baseline")
    
    # Add each file to the baseline
    for file_path in audio_files:
        logger.info(f"Adding to baseline: {os.path.basename(file_path)}")
        baseline.add_recording(file_path)
    
    # Build the baseline
    baseline.build()
    
    # Save the baseline if requested
    if args.save_baseline:
        os.makedirs(os.path.dirname(args.save_baseline), exist_ok=True)
        baseline.save(args.save_baseline)
        logger.info(f"Baseline saved to {args.save_baseline}")
    
    # Create analyzer with the baseline
    analyzer = SoundPoseAnalyzer(
        model_type=args.model,
        baseline=baseline,
        threshold=args.threshold,
    )
    
    # Analyze test file
    logger.info(f"Analyzing test file: {args.test_file}")
    results = analyzer.analyze_file(args.test_file)
    
    # Print summary
    logger.info(f"Analysis complete. Mean anomaly score: {results.get_mean_anomaly_score():.4f}")
    logger.info(f"Maximum anomaly score: {results.get_max_anomaly_score():.4f}")
    logger.info(f"Anomaly percentage: {results.get_anomaly_percentage():.2f}%")
    logger.info(f"Severity level: {results.get_severity_level()}")
    
    if results.has_anomalies():
        logger.info(f"Detected {len(results.get_anomaly_segments())} anomaly segments")
        
        for i, segment in enumerate(results.get_anomaly_segments()):
            logger.info(
                f"  Segment {i+1}: frames {segment['start']} - {segment['end']} "
                f"(score: {segment['score']:.4f})"
            )
    else:
        logger.info("No anomalies detected")
    
    # Save or display results
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        results.save_results(args.output)
        logger.info(f"Results saved to {args.output}")
        
        # Print diagnostic report
        report = results.get_diagnostic_report()
        report_path = os.path.join(args.output, "diagnostic_report.txt")
        
        with open(report_path, "w") as f:
            f.write(report)
        
        logger.info(f"Diagnostic report saved to {report_path}")
    else:
        # Display results
        results.plot_combined(title=f"Analysis Results - {os.path.basename(args.test_file)}")
        
        # Print diagnostic report
        print("\n" + "="*50 + "\n")
        print(results.get_diagnostic_report())
        print("\n" + "="*50)


if __name__ == "__main__":
    main()
