"""
Demo script for video processing.
"""

import argparse
from pathlib import Path
import sys
import json

from main import VideoPipeline

def main():
    """Run the demo script."""
    parser = argparse.ArgumentParser(description='Video Processing Demo')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--output', type=str, default='./data/results', help='Output directory for results')
    args = parser.parse_args()
    
    # Verify video path exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    print(f"Initializing video processing pipeline...")
    pipeline = VideoPipeline()
    
    # Process video
    print(f"Processing video: {video_path}")
    results = pipeline.process_video(str(video_path))
    
    # Save results
    result_file = output_dir / f"demo_results_{video_path.stem}.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processing complete! Results saved to: {result_file}")
    
    # Print summary
    print("\nProcessing Summary:")
    task_a = results.get('task_a', {})
    task_b = results.get('task_b', {})
    
    print("\nTask A - Image Analysis:")
    for provider, provider_results in task_a.get('detectors', {}).items():
        metrics = provider_results.get('metrics', {})
        print(f"  - {provider.upper()}:")
        print(f"    Latency: {metrics.get('avg_latency', 0):.2f} ms")
        print(f"    People detected: {metrics.get('total_people', 0)}")
        print(f"    Vehicles detected: {metrics.get('total_vehicles', 0)}")
    
    print("\nTask B - Object Tracking:")
    for provider, provider_results in task_b.get('trackers', {}).items():
        summary = provider_results.get('summary', {})
        print(f"  - {provider.upper()}:")
        print(f"    Processing time: {provider_results.get('processing_time', 0):.2f} s")
        print(f"    People tracked: {summary.get('people_tracked', 0)}")
        print(f"    Vehicles tracked: {summary.get('vehicles_tracked', 0)}")

if __name__ == "__main__":
    main() 