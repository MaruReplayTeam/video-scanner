# Video Scanner 🎥

![GitHub release (latest by date)](https://img.shields.io/github/v/release/MaruReplayTeam/video-scanner)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/MaruReplayTeam/video-scanner/build.yml)
![License](https://img.shields.io/badge/license-MIT-blue)

A powerful video quality analysis tool that detects frame drops, stuttering, and audio glitches in video files.

<div align="center">
  <img src="docs/logo.png" alt="BPCN LOGO" width="800"/>
</div>

## ✨ Features

- 🔍 Advanced frame analysis

  - Dropped frame detection
  - Frame stuttering analysis
  - FPS drop monitoring
  - Frame blending detection

- 🎵 Audio quality checks

  - Audio glitch detection
  - Audio loss analysis
  - Silent gap detection
  - Multi-channel support

- 🚀 Performance

  - GPU acceleration (NVIDIA)
  - Multi-threaded processing
  - Result caching
  - Progress indicators

- 💻 User Experience
  - Drag & drop support
  - Colorful terminal output
  - Timeline visualizations
  - Detailed analytics

## 🚀 Quick Start

### Installation

1. Download the latest release:

   ```bash
   # Option 1: Download executable
   Download VideoScanner.exe from Releases

   # Option 2: Install from source
   git clone https://github.com/MaruReplayTeam/video-scanner.git
   cd video-scanner
   pip install -r requirements.txt
   ```

### Usage

```bash
# Simple analysis
VideoScanner.exe video.mp4

# Command line options
python main.py [options] video_file

Options:
  --cpu            Force CPU mode
  --gpu-debug      Show GPU information
  --debug          Enable debug output
  --audio-sens N   Set audio sensitivity (0.1-5.0)
  --clear-cache    Clear analysis cache
  --ignore-cache   Force fresh analysis
```

## 📊 Analysis Types

### Frame Analysis

- **Dropped Frames**: Detects missing or duplicated frames
- **Stuttering**: Identifies irregular frame timing
- **FPS Drops**: Monitors frame rate consistency

### Audio Analysis

- **Glitches**: Detects audio artifacts and anomalies
- **Losses**: Identifies audio dropouts and gaps
- **Quality**: Measures audio consistency

## 🔧 Requirements

- Windows 10 or later
- For GPU acceleration:
  - NVIDIA GPU
  - CUDA compatible drivers
- Python 3.7+ (if running from source)

## 📝 Output Example

```
═══ VIDEO ANALYSIS REPORT ═══
File: example.mp4

▣ File Information:
  Resolution: 1920x1080
  FPS: 29.97
  Duration: 00:05:23

▣ Analysis Results:
  • 3 dropped frames detected
  • 2 stuttering events found
  • 1 FPS drop detected
  • 2 audio glitches identified
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

Made with ❤️ by Tama and MaruReplayTeam

## 🙏 Acknowledgments

- OpenCV for video processing
- MoviePy for audio analysis
- PyTorch for GPU acceleration
- Colorama for terminal colors
