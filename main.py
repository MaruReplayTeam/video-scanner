import sys
import os
import argparse
from video_analyzer import VideoAnalyzer
import cv2
from datetime import timedelta
import torch
from colorama import init, Fore, Back, Style
from pathlib import Path

# Initialize colorama
init()


def validate_video_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")

    valid_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    if not os.path.splitext(file_path)[1].lower() in valid_extensions:
        raise ValueError(
            f"Unsupported video format. Supported formats: {valid_extensions}"
        )


def generate_timeline_graph(duration, events, width=80):
    """Generate ASCII timeline graph of events"""
    timeline = [" "] * width
    if not events:
        return "|" + "".join(timeline) + "|"

    for event in events:
        # Convert timestamp to seconds if it's a string
        if isinstance(event["timestamp"], str):
            time_parts = [int(x) for x in event["timestamp"].split(":")]
            seconds = time_parts[0] * 3600 + time_parts[1] * 60 + time_parts[2]
        else:
            seconds = event["timestamp"]

        position = int((seconds / duration) * (width - 1))
        if 0 <= position < width:
            timeline[position] = "█"

    return "|" + "".join(timeline) + "|"


def format_file_info(analyzer):
    """Get detailed file information"""
    width = int(analyzer.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(analyzer.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = int(analyzer.cap.get(cv2.CAP_PROP_FOURCC))
    codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])

    return {
        "resolution": f"{width}x{height}",
        "codec": codec_str,
        "fps": round(analyzer.fps, 2),
        "duration": str(timedelta(seconds=int(analyzer.duration))),
        "frame_count": analyzer.frame_count,
        "size_mb": os.path.getsize(analyzer.video_path) / (1024 * 1024),
    }


def print_header():
    """Print a beautiful header with credits"""
    version = "1.0"
    header = f"""
{Fore.CYAN}
╔══════════════════════════════════════════════════════════════════════════╣
║     {Fore.GREEN}Video Quality Analyzer {Style.BRIGHT}v{version}{Style.RESET_ALL}{Fore.CYAN}                                        ║
║     {Fore.WHITE}Made with {Fore.RED}❤️{Fore.WHITE}  by {Fore.MAGENTA}Tama{Style.BRIGHT}@{Style.RESET_ALL}{Fore.CYAN}                                              ║
╚══════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(header)


def print_analysis_report(file_path, analysis_results, analyzer, debug=False):
    print(f"\n{Back.BLUE}{Fore.WHITE} VIDEO ANALYSIS REPORT {Style.RESET_ALL}")
    print(f"File: {Fore.CYAN}{os.path.basename(file_path)}{Style.RESET_ALL}")

    # Print detailed file information
    file_info = format_file_info(analyzer)
    print(f"\n{Fore.GREEN}▣ File Information:{Style.RESET_ALL}")
    print(f"  Resolution: {Fore.YELLOW}{file_info['resolution']}{Style.RESET_ALL}")
    print(f"  Codec: {Fore.YELLOW}{file_info['codec']}{Style.RESET_ALL}")
    print(f"  FPS: {Fore.YELLOW}{file_info['fps']}{Style.RESET_ALL}")
    print(f"  Duration: {Fore.YELLOW}{file_info['duration']}{Style.RESET_ALL}")
    print(f"  Frame Count: {Fore.YELLOW}{file_info['frame_count']}{Style.RESET_ALL}")
    print(f"  File Size: {Fore.YELLOW}{file_info['size_mb']:.2f} MB{Style.RESET_ALL}")

    print(f"\n{Fore.GREEN}▣ Dropped Frames Analysis:{Style.RESET_ALL}")
    dropped_frames = analysis_results["dropped_frames"]
    if dropped_frames:
        print(
            f"  Found {Fore.RED}{len(dropped_frames)}{Style.RESET_ALL} dropped frames:"
        )
        print("\n  Timeline Graph (each █ represents a drop):")
        print(
            f"  {Fore.RED}{generate_timeline_graph(analyzer.duration, dropped_frames)}{Style.RESET_ALL}"
        )
        print(f"  0{' ' * 37}{analyzer.duration:.1f}s")
        print("\n  Details:")
        for df in dropped_frames[:5]:
            print(
                f"  • Frame {df['frame']} at {df['timestamp']} (confidence: {df['confidence']:.2f})"
            )
        if len(dropped_frames) > 5:
            print(f"  • ... and {len(dropped_frames) - 5} more")
    else:
        print("  No dropped frames detected")

    print(f"\n{Fore.GREEN}▣ Stuttering Analysis:{Style.RESET_ALL}")
    stutter_events = analysis_results["stutter_events"]
    if stutter_events:
        print(f"  Detected {len(stutter_events)} stuttering events:")
        print("\n  Timeline Graph (each █ represents a stutter):")
        print(f"  {generate_timeline_graph(analyzer.duration, stutter_events)}")
        print(f"  0{' ' * 37}{analyzer.duration:.1f}s")
        print("\n  Details:")
        for se in stutter_events[:5]:
            print(f"  • Frame {se['frame']} at {se['timestamp']}")
            print(f"    Duration: {se['duration']:.2f}ms")
            print(f"    Severity: {se['severity']:.1f}x normal frame time")
        if len(stutter_events) > 5:
            print(f"  • ... and {len(stutter_events) - 5} more")
    else:
        print("  No stuttering detected")

    print(f"\n{Fore.GREEN}▣ FPS Analysis:{Style.RESET_ALL}")
    fps_drops = analysis_results["fps_drops"]
    if fps_drops:
        print(f"  Detected {len(fps_drops)} FPS drop events:")
        print("\n  Timeline Graph (each █ represents an FPS drop):")
        print(f"  {generate_timeline_graph(analyzer.duration, fps_drops)}")
        print(f"  0{' ' * 37}{analyzer.duration:.1f}s")
        print("\n  Details:")
        for drop in fps_drops[:5]:
            print(f"  • At {drop['timestamp']} (Frame {drop['frame']})")
            print(
                f"    Current FPS: {drop['current_fps']:.1f} (Target: {analyzer.target_fps:.1f})"
            )
            print(f"    Duration: {drop['duration']:.0f}ms")
            print(f"    Severity: {drop['severity']*100:.1f}% below target")
        if len(fps_drops) > 5:
            print(f"  • ... and {len(fps_drops) - 5} more")
    else:
        print("  No significant FPS drops detected")

    print(f"\n{Fore.GREEN}▣ Audio Analysis:{Style.RESET_ALL}")
    audio_glitches = analysis_results["audio_glitches"]

    # Separate and filter audio issues
    losses = [
        g
        for g in audio_glitches
        if g["type"] == "loss" and g["severity"] > analyzer.min_loss_severity
    ]
    glitches = [
        g for g in audio_glitches if g["type"] == "glitch" and g["severity"] > 0.01
    ]

    if losses or glitches:
        # Report audio losses
        if losses:
            print(f"  Found {len(losses)} significant audio losses:")
            for loss in losses[:5]:
                severity_color = (
                    Fore.RED
                    if loss["severity"] > 1.0
                    else Fore.YELLOW if loss["severity"] > 0.5 else Fore.WHITE
                )
                print(f"  • At {loss['timestamp']}")
                print(
                    f"    Duration: {severity_color}{loss['duration_ms']:.1f}ms{Style.RESET_ALL}"
                )
                print(
                    f"    Severity: {severity_color}{loss['severity']:.1f}x frame time{Style.RESET_ALL}"
                )
                if "samples" in loss:
                    print(
                        f"    Samples: {loss['samples']} ({loss['duration_ms']/1000*analyzer.fps:.1f} frames)"
                    )
            if len(losses) > 5:
                print(f"  • ... and {len(losses) - 5} more losses")

        # Report audio glitches
        if glitches:
            print(
                f"\n  Found {len(glitches)} audio glitches (sensitivity: {analyzer.audio_sensitivity}x):"
            )
            for ag in glitches[:5]:
                severity_color = (
                    Fore.RED
                    if ag["severity"] > 2.0
                    else Fore.YELLOW if ag["severity"] > 1.5 else Fore.WHITE
                )
                print(f"  • At {ag['timestamp']}")
                print(
                    f"    Amplitude: {severity_color}{ag['amplitude']:.2f}{Style.RESET_ALL}"
                )
                print(
                    f"    Severity: {severity_color}{ag['severity']:.1f}x threshold{Style.RESET_ALL}"
                )
            if len(glitches) > 5:
                print(f"  • ... and {len(glitches) - 5} more glitches")
    else:
        print("  No significant audio issues detected")


def verify_gpu_support(debug=False):
    """Verify GPU support and print debug information"""
    if torch.cuda.is_available():
        print(f"\n{Fore.GREEN}GPU Information:{Style.RESET_ALL}")
        if debug:
            print(f"CUDA Version: {Fore.CYAN}{torch.version.cuda}{Style.RESET_ALL}")
            print(
                f"GPU Device: {Fore.CYAN}{torch.cuda.get_device_name(0)}{Style.RESET_ALL}"
            )
            print(f"GPU Count: {Fore.CYAN}{torch.cuda.device_count()}{Style.RESET_ALL}")
        else:
            print(f"Using: {Fore.CYAN}{torch.cuda.get_device_name(0)}{Style.RESET_ALL}")
        return True
    else:
        print(f"\n{Fore.RED}No GPU Support Found{Style.RESET_ALL}")
        if debug:
            print(f"{Fore.YELLOW}- Please ensure NVIDIA drivers are installed")
            print("- Check that PyTorch is installed with CUDA support")
            print(f"- Run 'nvidia-smi' to verify GPU is detected{Style.RESET_ALL}")
        return False


def clear_cache():
    """Clear all cached analysis results"""
    cache_dir = Path(__file__).parent / "cache"
    if cache_dir.exists():
        for cache_file in cache_dir.glob("*.json"):
            cache_file.unlink()
        print(f"{Fore.GREEN}Cache cleared successfully{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}No cache directory found{Style.RESET_ALL}")


def get_dropped_file():
    """Get file path from drag & drop or command line"""
    if len(sys.argv) > 1:
        return sys.argv[1]
    return None


def create_file_association(debug=False):
    """Create file associations for video files"""
    import winreg

    try:
        executable = sys.executable if getattr(sys, "frozen", False) else __file__

        # File types to associate
        extensions = [".mp4", ".avi", ".mov", ".mkv"]

        for ext in extensions:
            # Create file type
            with winreg.CreateKey(
                winreg.HKEY_CURRENT_USER, f"Software\\Classes\\{ext}"
            ) as key:
                winreg.SetValue(key, "", winreg.REG_SZ, "VideoScanner.Video")

            # Create command
            with winreg.CreateKey(
                winreg.HKEY_CURRENT_USER,
                "Software\\Classes\\VideoScanner.Video\\shell\\open\\command",
            ) as key:
                winreg.SetValue(key, "", winreg.REG_SZ, f'"{executable}" "%1"')

    except Exception as e:
        if debug:
            print(f"Warning: Could not create file association: {e}")


def get_base_path():
    """Get base path for resources, handling both development and frozen env"""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).parent


def main():
    # Set application path
    os.chdir(get_base_path())

    # Get file from drag & drop
    video_file = get_dropped_file()

    # Parse additional arguments
    parser = argparse.ArgumentParser(description="Video quality analysis tool")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument(
        "--gpu-debug", action="store_true", help="Show GPU debug information"
    )
    parser.add_argument("--debug", action="store_true", help="Show debug information")
    parser.add_argument(
        "--audio-sens",
        type=float,
        default=1.0,
        help="Audio glitch detection sensitivity (0.1-5.0, default: 1.0)",
    )
    parser.add_argument(
        "--clear-cache", action="store_true", help="Clear cached analysis results"
    )
    parser.add_argument(
        "--ignore-cache",
        action="store_true",
        help="Ignore cached results and reanalyze",
    )

    # Parse only known args to allow for drag & drop
    args, unknown = parser.parse_known_args()

    if getattr(sys, "frozen", False):
        # Create file association when running as exe
        create_file_association(debug=args.debug)

    if args.clear_cache:
        clear_cache()
        return

    print_header()

    # Handle GPU debug mode first
    if args.gpu_debug:
        verify_gpu_support(debug=True)
        return

    # Check for required video file argument for normal operation
    if not video_file:
        parser.error("video_file is required unless using --gpu-debug")

    try:
        validate_video_file(video_file)
        use_gpu = not args.cpu

        if use_gpu:
            verify_gpu_support(debug=args.debug)

        analyzer = VideoAnalyzer(
            video_file,
            use_gpu=use_gpu,
            debug=args.debug,
            audio_sensitivity=args.audio_sens,
        )

        if args.ignore_cache:
            # Force reanalysis by deleting existing cache
            cache_path = analyzer._get_cache_path()
            if cache_path.exists():
                cache_path.unlink()
                if args.debug:
                    print("Ignoring cache and performing fresh analysis")

        # Get comprehensive analysis
        analysis_results = analyzer.analyze_video_quality()

        # Print report with all results
        print_analysis_report(video_file, analysis_results, analyzer, debug=args.debug)

    except Exception as e:
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        if args.debug:
            import traceback

            print(f"\n{Fore.YELLOW}Debug traceback:{Style.RESET_ALL}")
            traceback.print_exc()
        sys.exit(1)

    # Wait for input if running as exe
    if getattr(sys, "frozen", False):
        input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
