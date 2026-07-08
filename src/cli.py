"""Command-line interface for mono3d-recon."""

import argparse
import json
import sys
from pathlib import Path

from .config import PipelineConfig
from .pipeline import ReconstructionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="3D Scene Reconstruction from Monocular Video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic reconstruction
  python -m src.cli reconstruct --input video.mp4

  # High quality with custom output
  python -m src.cli reconstruct --input video.mp4 --output my_scene --quality high

  # Use specific depth model on GPU
  python -m src.cli reconstruct --input video.mp4 --depth-model midas --device cuda

  # Just extract frames
  python -m src.cli extract --input video.mp4 --fps 3

  # Get video info
  python -m src.cli info --input video.mp4
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- reconstruct ---
    recon_parser = subparsers.add_parser(
        "reconstruct", help="Full reconstruction pipeline"
    )
    recon_parser.add_argument(
        "--input", "-i", required=True, help="Path to input video"
    )
    recon_parser.add_argument(
        "--output", "-o", default="outputs", help="Output directory"
    )
    recon_parser.add_argument(
        "--quality", "-q", choices=["low", "medium", "high"],
        default="medium", help="Quality preset"
    )
    recon_parser.add_argument(
        "--depth-model", choices=["midas", "dav2"],
        default="midas", help="Depth estimation model"
    )
    recon_parser.add_argument(
        "--mesh-method", choices=["poisson", "bpa"],
        default="poisson", help="Mesh reconstruction method"
    )
    recon_parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"],
        default="auto", help="Compute device"
    )
    recon_parser.add_argument(
        "--fps", type=float, default=None, help="Override frame extraction FPS"
    )
    recon_parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # --- extract ---
    extract_parser = subparsers.add_parser(
        "extract", help="Extract frames only"
    )
    extract_parser.add_argument("--input", "-i", required=True)
    extract_parser.add_argument("--output", "-o", default="data/frames")
    extract_parser.add_argument("--fps", type=float, default=2.0)
    extract_parser.add_argument("--min-sharpness", type=float, default=50.0)

    # --- info ---
    info_parser = subparsers.add_parser("info", help="Show video information")
    info_parser.add_argument("--input", "-i", required=True)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "info":
        from .video_utils import get_video_info
        info = get_video_info(args.input)
        print(json.dumps(info, indent=2))

    elif args.command == "extract":
        from .video_utils import extract_frames
        from .config import VideoConfig
        vc = VideoConfig(fps=args.fps, min_sharpness=args.min_sharpness)
        paths = extract_frames(args.input, args.output, vc)
        print(f"\nExtracted {len(paths)} frames")

    elif args.command == "reconstruct":
        # Build config
        if args.config:
            config = PipelineConfig.load(args.config)
        else:
            config = PipelineConfig()

        config.quality = args.quality
        config.depth.model = args.depth_model
        config.depth.device = args.device
        config.mesh.method = args.mesh_method

        if args.fps is not None:
            config.video.fps = args.fps

        # Run pipeline
        pipeline = ReconstructionPipeline(config=config)
        stats = pipeline.run(args.input, args.output)

        print("\n" + "=" * 50)
        print("RECONSTRUCTION COMPLETE")
        print("=" * 50)
        print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
