"""
Basic analysis example for SoundPose.

This example demonstrates how to use SoundPose to analyze an audio file
and visualize the results.
"""

import os
import sys
import argparse
import logging

# Add parent directory to the path to import SoundPose
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from soundpose import SoundPoseAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Run the basic analysis example."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SoundPose Basic Analysis Example")
    parser.add_argument(
        "--file", "-f", type=str, required=True, help="Path to the audio file to analyze"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.75, help="Threshold for anomaly detection"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="voice", choices=["voice", "mechanical"],
        help="Type of model to use"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Directory to save the results (if None, results are displayed)"
    )
    
    args = parser.parse_args()
    
    # Check if the file exists
    if not os.path.isfile(args.file):
        logger.error(f"File not found: {args.file}")
        return
    
    logger.info(f"Analyzing file: {args.file}")
    logger.info(f"Using model: {args.model}")
    logger.info(f"Using threshold: {args.threshold}")
    
    # Create analyzer
    analyzer = SoundPoseAnalyzer(
        model_type=args.model,
        threshold=args.threshold,
    )
    
    # Analyze file
    results = analyzer.analyze_file(args.file)
    
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
        results.plot_combined(title=f"Analysis Results - {os.path.basename(args.file)}")
        
        # Print diagnostic report
        print("\n" + "="*50 + "\n")
        print(results.get_diagnostic_report())
        print("\n" + "="*50)


if __name__ == "__main__":
    main()
