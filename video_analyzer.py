import cv2
import numpy as np
import torch
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path

try:
    from moviepy import VideoFileClip
except ImportError:
    raise ImportError(
        "MoviePy is required but not installed. Please install it using 'pip install moviepy'"
    )
from tqdm import tqdm


class VideoAnalyzer:
    def __init__(self, video_path, use_gpu=True, debug=False, audio_sensitivity=2.0):
        self.video_path = video_path
        self.debug = debug
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
        self.blend_threshold = 0.85  # Threshold for frame blend detection
        self.diff_threshold = 1.5  # Threshold for frame difference
        self.target_fps = self.fps  # Store target FPS
        self.fps_drop_threshold = 0.8  # Detect drops below 80% of target FPS

        # Enhanced GPU verification
        self.use_gpu = use_gpu
        if use_gpu:
            if not torch.cuda.is_available():
                print("WARNING: GPU requested but CUDA is not available")
                print("CUDA Status:", torch.cuda.is_available())
                print("GPU Count:", torch.cuda.device_count())
                print(
                    "GPU Device:",
                    (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.is_available()
                        else "None"
                    ),
                )
                self.use_gpu = False
            else:
                print("GPU Mode Active:")
                print(f"- Using: {torch.cuda.get_device_name(0)}")
                print(
                    f"- Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB"
                )
                self.device = torch.device("cuda")
                torch.backends.cudnn.benchmark = True

        if not self.use_gpu:
            print("Using CPU mode")
            self.device = torch.device("cpu")

        self.audio_sensitivity = max(
            0.1, min(5.0, audio_sensitivity)
        )  # Clamp between 0.1 and 5.0
        if debug:
            print(f"Audio sensitivity: {self.audio_sensitivity}x")
        self.audio_gap_threshold = 0.05  # 50ms gap threshold
        self.min_loss_severity = 0.1  # Minimum severity for audio losses

        self.cache_dir = Path(__file__).parent / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.video_hash = self._calculate_video_hash()
        if debug:
            print(f"Video hash: {self.video_hash}")

    def _calculate_video_hash(self):
        """Calculate a hash of the video file for caching"""
        hasher = hashlib.sha256()
        with open(self.video_path, 'rb') as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _get_cache_path(self):
        """Get the cache file path for current video"""
        return self.cache_dir / f"{self.video_hash}.json"

    def _load_from_cache(self):
        """Try to load analysis results from cache"""
        cache_path = self._get_cache_path()
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                if self.debug:
                    print(f"Loaded results from cache: {cache_path}")
                return cached_data
            except Exception as e:
                if self.debug:
                    print(f"Failed to load cache: {e}")
        return None

    def _save_to_cache(self, results):
        """Save analysis results to cache"""
        try:
            cache_path = self._get_cache_path()
            with open(cache_path, 'w') as f:
                json.dump(results, f)
            if self.debug:
                print(f"Saved results to cache: {cache_path}")
        except Exception as e:
            if self.debug:
                print(f"Failed to save cache: {e}")

    def _to_gpu(self, frame):
        """Convert frame to GPU tensor if GPU is enabled"""
        if self.use_gpu:
            return torch.from_numpy(frame).cuda()
        return frame

    def _calculate_frame_similarity(self, frame1, frame2):
        """Calculate similarity between frames using GPU if available"""
        try:
            if self.use_gpu:
                # Convert frames to GPU tensors
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                gpu_gray1 = torch.from_numpy(gray1).to(self.device).float()
                gpu_gray2 = torch.from_numpy(gray2).to(self.device).float()

                # Updated autocast syntax
                with torch.amp.autocast("cuda"):
                    # Calculate difference on GPU
                    diff = torch.abs(gpu_gray1 - gpu_gray2)
                    diff_mean = float(diff.mean().cpu().numpy())

                    # Histogram calculation on GPU
                    hist1 = torch.histc(gpu_gray1, bins=256, min=0, max=255)
                    hist2 = torch.histc(gpu_gray2, bins=256, min=0, max=255)

                    # Normalize histograms
                    hist1 = hist1 / hist1.sum()
                    hist2 = hist2 / hist2.sum()

                    # Calculate correlation
                    hist_correlation = float(
                        torch.sum(torch.sqrt(hist1 * hist2)).cpu().numpy()
                    )

                    # Calculate SSIM-like metric on GPU
                    ssim = float(
                        torch.nn.functional.cosine_similarity(
                            gpu_gray1.flatten().unsqueeze(0),
                            gpu_gray2.flatten().unsqueeze(0),
                        )
                        .cpu()
                        .numpy()
                    )

                return {
                    "hist_correlation": hist_correlation,
                    "ssim": ssim,
                    "diff_mean": diff_mean,
                }
        except RuntimeError as e:
            print(f"GPU Error: {e}")
            print("Falling back to CPU mode")
            self.use_gpu = False
            return self._calculate_frame_similarity_cpu(frame1, frame2)

    def _calculate_frame_similarity_cpu(self, frame1, frame2):
        """CPU fallback for frame similarity calculation"""
        # Existing CPU implementation
        # Convert frames to grayscale for histogram comparison
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate histogram correlation
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        hist_correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # Calculate mean structural similarity
        ssim = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)[0][0]

        # Calculate frame difference
        diff = cv2.absdiff(frame1, frame2)
        diff_mean = np.mean(diff)

        return {
            "hist_correlation": hist_correlation,
            "ssim": ssim,
            "diff_mean": diff_mean,
        }

    def analyze_dropped_frames(self):
        """GPU-accelerated dropped frame analysis"""
        if self.use_gpu:
            torch.cuda.empty_cache()

        dropped_frames = []
        previous_frame = None
        frame_buffer = []  # Buffer for detecting frame blending

        for frame_num in tqdm(range(self.frame_count), desc="Analyzing dropped frames"):
            ret, frame = self.cap.read()
            if not ret:
                break

            if previous_frame is not None:
                metrics = self._calculate_frame_similarity(previous_frame, frame)

                # Detect frame drops considering blending
                is_dropped = (
                    metrics["hist_correlation"] > self.blend_threshold
                    and metrics["ssim"] > self.blend_threshold
                    and metrics["diff_mean"] < self.diff_threshold
                )

                # Check for frame blending patterns
                if len(frame_buffer) >= 2:
                    blend_check = self._calculate_frame_similarity(
                        frame_buffer[-2], frame
                    )
                    is_blend = blend_check["hist_correlation"] > 0.7

                    if is_dropped and not is_blend:
                        timestamp = frame_num / self.fps
                        dropped_frames.append(
                            {
                                "frame": frame_num,
                                "timestamp": self._format_timestamp(timestamp),
                                "confidence": float(metrics["hist_correlation"]),
                                "type": "dropped",
                            }
                        )

            # Update frame buffer
            frame_buffer.append(frame.copy())
            if len(frame_buffer) > 3:
                frame_buffer.pop(0)
            previous_frame = frame.copy()

        if self.use_gpu:
            torch.cuda.empty_cache()

        return dropped_frames

    def analyze_stuttering(self):
        """Analyze video for stuttering by tracking frame delivery times"""
        stutter_events = []
        frame_times = []

        # Reset video position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Get start time
        start_time = None
        prev_frame_time = None
        expected_frame_time = 1.0 / self.fps

        for frame_num in tqdm(range(self.frame_count), desc="Analyzing stuttering"):
            ret, _ = self.cap.read()
            if not ret:
                break

            # Get current frame timestamp in milliseconds
            current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)

            if start_time is None:
                start_time = current_time
                prev_frame_time = current_time
                continue

            # Calculate actual frame time
            frame_time = (current_time - prev_frame_time) / 1000.0  # Convert to seconds
            frame_times.append(frame_time)

            # Detect stutter if frame time is significantly longer than expected
            if frame_time > (expected_frame_time * 1.5):  # 50% longer than expected
                stutter_events.append(
                    {
                        "timestamp": self._format_timestamp(
                            (current_time - start_time) / 1000.0
                        ),
                        "duration": frame_time * 1000,  # Convert to milliseconds
                        "frame": frame_num,
                        "severity": frame_time / expected_frame_time,
                    }
                )

            prev_frame_time = current_time

        # Reset video position for other operations
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return stutter_events

    def analyze_fps_drops(self, window_size=30):
        """Analyze FPS drops using a sliding window"""
        fps_drops = []
        frame_times = []
        start_time = None
        prev_time = None

        # Reset video position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for frame_num in tqdm(range(self.frame_count), desc="Analyzing FPS drops"):
            ret, _ = self.cap.read()
            if not ret:
                break

            current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            if start_time is None:
                start_time = current_time
                prev_time = current_time
                continue

            if prev_time is not None:
                frame_time = current_time - prev_time
                frame_times.append(frame_time)

                # Analysis using sliding window
                if len(frame_times) >= window_size:
                    window_duration = sum(frame_times[-window_size:])
                    current_fps = window_size / window_duration

                    if current_fps < (self.target_fps * self.fps_drop_threshold):
                        fps_drops.append(
                            {
                                "timestamp": self._format_timestamp(
                                    current_time - start_time
                                ),
                                "frame": frame_num,
                                "current_fps": current_fps,
                                "duration": window_duration * 1000,  # ms
                                "severity": 1 - (current_fps / self.target_fps),
                            }
                        )

            prev_time = current_time

        # Reset video position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return fps_drops

    def analyze_audio_glitches(self):
        """Analyze audio for glitches and short losses"""
        glitches = []
        video = None
        try:
            video = VideoFileClip(self.video_path)
            if video.audio is None:
                print(f"Warning: No audio track found in video")
                return glitches

            try:
                # Get audio data with proper sequence conversion
                raw_audio = video.audio.to_soundarray()

                if self.debug:
                    print(f"Debug: Raw audio type: {type(raw_audio)}")
                    print(
                        f"Debug: Raw audio shape before processing: {audio_array.shape}"
                    )

                # Convert to numpy array safely
                if isinstance(raw_audio, (list, tuple)):
                    audio_array = np.stack(raw_audio)
                else:
                    audio_array = np.array(raw_audio)

                # Ensure 2D array for consistent processing
                if audio_array.ndim == 1:
                    audio_array = audio_array.reshape(-1, 1)

                # Convert stereo to mono by averaging all channels
                processed_audio = np.mean(audio_array, axis=-1)

                # Normalize audio
                abs_max = np.abs(processed_audio).max()
                if abs_max > 0:
                    processed_audio = processed_audio / abs_max

                if self.debug:
                    print(
                        f"Debug: Final processed audio shape: {processed_audio.shape}"
                    )

                # Analysis parameters with sensitivity adjustment
                base_threshold = 2.0
                threshold = (
                    base_threshold / self.audio_sensitivity
                )  # Lower threshold = more sensitive
                sample_rate = int(video.audio.fps)
                window_size = int(
                    sample_rate / (5 * self.audio_sensitivity)
                )  # Adjust window size with sensitivity
                min_gap_samples = int(sample_rate * 0.001)  # 1ms minimum gap
                max_gap_samples = int(sample_rate * self.audio_gap_threshold)  # max gap length

                if self.debug:
                    print(f"Audio analysis parameters:")
                    print(f"- Base threshold: {base_threshold}")
                    print(f"- Adjusted threshold: {threshold}")
                    print(
                        f"- Window size: {window_size} samples ({window_size/sample_rate*1000:.1f}ms)"
                    )
                    print(f"- Gap detection range: {min_gap_samples}-{max_gap_samples} samples ({min_gap_samples/sample_rate*1000:.1f}-{max_gap_samples/sample_rate*1000:.1f}ms)")

                # Detect silence gaps (potential audio loss)
                silence_threshold = 0.01
                significant_gap_threshold = 0.05  # 5% of frame time
                in_gap = False
                gap_start = 0
                gap_length = 0
                consecutive_silent = 0

                for i in tqdm(range(window_size, len(processed_audio)), desc="Analyzing audio"):
                    window = processed_audio[i - window_size : i]
                    current = processed_audio[i]
                    
                    # Enhanced gap detection
                    if abs(current) < silence_threshold:
                        consecutive_silent += 1
                        if not in_gap and consecutive_silent >= 3:  # Require 3 consecutive silent samples
                            gap_start = i - consecutive_silent
                            in_gap = True
                        gap_length += 1
                    else:
                        if in_gap and min_gap_samples <= gap_length <= max_gap_samples:
                            time = gap_start / sample_rate
                            duration_ms = (gap_length / sample_rate) * 1000
                            severity = duration_ms / (1000/self.fps)  # Relative to frame duration
                            
                            # Only add significant gaps
                            if severity > self.min_loss_severity and duration_ms > (1000/self.fps * significant_gap_threshold):
                                glitches.append({
                                    "timestamp": self._format_timestamp(time),
                                    "amplitude": 0.0,
                                    "position": gap_start,
                                    "time_seconds": time,
                                    "duration_ms": duration_ms,
                                    "type": "loss",
                                    "severity": severity,
                                    "samples": gap_length
                                })
                        in_gap = False
                        gap_length = 0
                        consecutive_silent = 0

                    # Regular amplitude glitch detection
                    amplitude_change = np.abs(current - np.mean(window)) * self.audio_sensitivity
                    if amplitude_change > threshold:
                        severity = amplitude_change / threshold
                        if severity > 0.0:  # Only add if severity > 0
                            time = i / sample_rate
                            glitches.append({
                                "timestamp": self._format_timestamp(time),
                                "amplitude": float(amplitude_change),
                                "position": i,
                                "time_seconds": time,
                                "type": "glitch",
                                "severity": severity
                            })

                # Filter and sort glitches
                losses = [g for g in glitches if g["type"] == "loss" and g["severity"] > self.min_loss_severity]
                glitches = [g for g in glitches if g["type"] == "glitch" and g["severity"] > 0.0]
                
                # Merge nearby losses
                merged_losses = []
                if losses:
                    current_loss = losses[0]
                    for loss in losses[1:]:
                        gap_between = loss["position"] - (current_loss["position"] + current_loss["samples"])
                        if gap_between < sample_rate * 0.1:  # Merge if gap is less than 100ms
                            # Merge the losses
                            current_loss["duration_ms"] += loss["duration_ms"]
                            current_loss["samples"] += loss["samples"] + gap_between
                            current_loss["severity"] = max(current_loss["severity"], loss["severity"])
                        else:
                            merged_losses.append(current_loss)
                            current_loss = loss
                    merged_losses.append(current_loss)
                
                # Combine filtered glitches and merged losses
                glitches.extend(merged_losses)
                glitches.sort(key=lambda x: x["position"])

            except Exception as e:
                print(f"Warning: Audio processing error - {str(e)}")
                if self.debug:
                    print(f"Debug info:")
                    print(
                        f"- Audio clip duration: {video.audio.duration if video.audio else 'None'}"
                    )
                    print(f"- Audio fps: {video.audio.fps if video.audio else 'None'}")
                    if "raw_audio" in locals():
                        print(f"- Raw audio type: {type(raw_audio)}")
                    if "audio_array" in locals():
                        print(f"- Audio array shape: {audio_array.shape}")
                return []

        except Exception as e:
            print(f"Warning: Audio loading failed - {str(e)}")
            return []

        finally:
            if video is not None:
                video.close()

        return glitches

    def analyze_video_quality(self):
        """Run analysis with caching support"""
        # Try to load from cache first
        cached_results = self._load_from_cache()
        if cached_results is not None:
            return cached_results

        if self.use_gpu:
            torch.cuda.empty_cache()

        results = {
            "dropped_frames": self.analyze_dropped_frames(),
            "stutter_events": self.analyze_stuttering(),
            "fps_drops": self.analyze_fps_drops(),
            "audio_glitches": self.analyze_audio_glitches(),
            "metadata": {
                "video_hash": self.video_hash,
                "duration": self.duration,
                "fps": self.fps,
                "frame_count": self.frame_count,
                "analysis_version": "1.0"
            }
        }

        if self.use_gpu:
            torch.cuda.empty_cache()

        # Save results to cache
        self._save_to_cache(results)
        
        return results

    def _format_timestamp(self, seconds):
        return str(timedelta(seconds=int(seconds)))

    def __del__(self):
        if hasattr(self, "cap"):
            self.cap.release()
