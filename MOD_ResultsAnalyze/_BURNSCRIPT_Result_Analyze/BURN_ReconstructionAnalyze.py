#!/usr/bin/env python
"""
BURN_ReconstructionAnalyze - Analyze COLMAP/GLOMAP Reconstruction Results
========================================================================
This script loads reconstructions from mapping results and generates 
HTML and JSON reports with statistics and camera trajectory information.

Features:
- Recursively finds all reconstructions (sparse/0, sparse/1, etc.)
- Counts actual images from the images folder for accurate registration ratio
- Can select best model from multiple models (--best-only)
- Exports to both HTML and JSON

Usage:
    python BURN_ReconstructionAnalyze.py <recon_path1> [recon_path2] ... [--output <output_dir>]

Example:
    python BURN_ReconstructionAnalyze.py "F:\\IMAGEs\\17\\mapping3r_individual_20260123_182327" --best-only
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from UTILS4BS_ReconstructionAnalyze import (
    find_all_reconstructions,
    load_reconstruction,
    load_multiple_reconstructions,
    select_best_models,
    export_analysis_json,
    generate_html_report,
    export_trajectories_json,
)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze COLMAP/GLOMAP reconstruction results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python BURN_ReconstructionAnalyze.py "F:\\IMAGEs\\17\\mapping3r_individual_20260123_182327"
  python BURN_ReconstructionAnalyze.py path1 path2 --best-only --output ./analysis_results
        """
    )
    parser.add_argument(
        'recon_paths',
        nargs='+',
        help='Path(s) to reconstruction folder(s)'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output directory for reports (default: first recon path parent)'
    )
    parser.add_argument(
        '--best-only',
        action='store_true',
        help='Only keep best model per location (100%% registered or most images)'
    )
    parser.add_argument(
        '--export-json',
        action='store_true',
        help='Also export analysis as JSON file'
    )
    parser.add_argument(
        '--export-trajectories',
        action='store_true',
        help='Also export camera trajectories as JSON files'
    )
    parser.add_argument(
        '--title',
        default=None,
        help='Custom title for the HTML report'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not recursively search for reconstructions (default: recursive)'
    )
    
    args = parser.parse_args()
    
    # Convert paths
    recon_paths = [Path(p) for p in args.recon_paths]
    
    # Validate paths
    valid_paths = []
    for p in recon_paths:
        if p.exists():
            valid_paths.append(p)
            print(f"[OK] Found: {p}")
        else:
            print(f"[!!] Not found: {p}")
    
    if not valid_paths:
        print("\nError: No valid reconstruction paths found!")
        sys.exit(1)
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = valid_paths[0].parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load reconstructions
    print(f"\n{'='*60}")
    print("Loading reconstructions...")
    print(f"Recursive search: {not args.no_recursive}")
    print(f"{'='*60}")
    
    results = load_multiple_reconstructions(valid_paths, recursive=not args.no_recursive)
    
    # Filter out failed loads
    successful = {k: v for k, v in results.items() if v[1] is not None}
    
    if not successful:
        print("\nError: Failed to load any reconstructions!")
        sys.exit(1)
    
    # Select best models if requested
    if args.best_only:
        print(f"\n{'='*60}")
        print("Selecting best models...")
        print(f"{'='*60}")
        all_count = len(successful)
        successful = select_best_models(successful)
        print(f"  Selected {len(successful)} best models from {all_count} total")
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Export JSON analysis
    if args.export_json:
        print(f"\n{'='*60}")
        print("Exporting analysis to JSON...")
        print(f"{'='*60}")
        json_path = output_dir / f"reconstruction_analysis_{timestamp}.json"
        export_analysis_json(successful, json_path)
    
    # Generate HTML report
    print(f"\n{'='*60}")
    print("Generating HTML report...")
    print(f"{'='*60}")
    
    report_name = f"reconstruction_analysis_{timestamp}.html"
    report_path = output_dir / report_name
    
    title = args.title or f"Reconstruction Analysis ({len(successful)} datasets)"
    generate_html_report(successful, report_path, title=title)
    
    # Export trajectories if requested
    if args.export_trajectories:
        print(f"\n{'='*60}")
        print("Exporting trajectories to JSON...")
        print(f"{'='*60}")
        
        for name, (trajectories, stats) in successful.items():
            if trajectories:
                safe_name = name.replace('/', '_').replace('\\', '_')
                json_path = output_dir / f"trajectories_{safe_name}_{timestamp}.json"
                export_trajectories_json(trajectories, json_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Reconstructions analyzed: {len(successful)}")
    print(f"HTML report: {report_path}")
    
    # Print quick stats
    print(f"\nQuick Summary:")
    print("-" * 40)
    for name, (trajs, stats) in successful.items():
        ratio = stats.registration_ratio * 100
        status = "[OK]" if ratio > 99 else "[~~]" if ratio > 50 else "[!!]"
        print(f"  {status} {name}:")
        print(f"      Images: {stats.num_images_registered}/{stats.num_images_total} ({ratio:.1f}%)")
        print(f"      Points: {stats.num_points3D:,}")
        print(f"      Reproj: {stats.mean_reproj_error:.3f} px")
    
    return report_path


if __name__ == '__main__':
    main()
