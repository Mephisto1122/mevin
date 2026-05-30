"""
Split a CCTV compilation video into individual clips.
Uses ffmpeg scene detection to find cut points.

Usage:
  python split_clips.py cctv_compilation.mp4
  python split_clips.py cctv_compilation.mp4 --threshold 0.3 --min-duration 3
"""
import subprocess, sys, os, re

def detect_scenes(video_path, threshold=0.35):
    """Use ffmpeg to detect scene changes."""
    print(f"Detecting scenes in {video_path} (threshold={threshold})...")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse timestamps from showinfo output
    times = [0.0]
    for line in result.stderr.split('\n'):
        if 'showinfo' in line and 'pts_time:' in line:
            match = re.search(r'pts_time:\s*([\d.]+)', line)
            if match:
                t = float(match.group(1))
                times.append(t)
    
    # Get total duration
    for line in result.stderr.split('\n'):
        if 'Duration:' in line:
            match = re.search(r'Duration:\s*(\d+):(\d+):(\d+\.\d+)', line)
            if match:
                h, m, s = float(match.group(1)), float(match.group(2)), float(match.group(3))
                total = h * 3600 + m * 60 + s
                times.append(total)
                break
    
    return sorted(set(times))

def split_video(video_path, times, output_dir="clips", min_duration=3, max_duration=120):
    """Split video at detected scene times."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out very short segments
    segments = []
    for i in range(len(times) - 1):
        start = times[i]
        end = times[i + 1]
        duration = end - start
        if duration >= min_duration and duration <= max_duration:
            segments.append((start, end, duration))
    
    # Merge very short consecutive segments
    merged = []
    i = 0
    while i < len(segments):
        start, end, dur = segments[i]
        # If this segment is short, try to merge with next
        while dur < min_duration and i + 1 < len(segments):
            i += 1
            end = segments[i][1]
            dur = end - start
        merged.append((start, end, end - start))
        i += 1
    
    print(f"\nFound {len(merged)} clips (min {min_duration}s, max {max_duration}s)")
    print("-" * 50)
    
    for i, (start, end, duration) in enumerate(merged):
        clip_name = f"clip_{i+1:03d}.mp4"
        clip_path = os.path.join(output_dir, clip_name)
        
        print(f"  [{i+1}/{len(merged)}] {start:.1f}s - {end:.1f}s ({duration:.1f}s) -> {clip_name}")
        
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-an",  # no audio needed for CCTV
            "-loglevel", "error",
            clip_path
        ]
        subprocess.run(cmd)
    
    print(f"\nDone! {len(merged)} clips saved to {output_dir}/")
    print(f"Load them in Mevin: Open Video File -> select from {output_dir}/")

if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "cctv_compilation.mp4"
    threshold = 0.35
    min_dur = 3
    
    # Parse optional args
    for i, arg in enumerate(sys.argv):
        if arg == "--threshold" and i + 1 < len(sys.argv):
            threshold = float(sys.argv[i + 1])
        if arg == "--min-duration" and i + 1 < len(sys.argv):
            min_dur = int(sys.argv[i + 1])
    
    if not os.path.exists(video):
        print(f"File not found: {video}")
        print(f"Download first: yt-dlp -f worst[ext=mp4] -o {video} URL")
        sys.exit(1)
    
    times = detect_scenes(video, threshold)
    print(f"Found {len(times)} scene changes")
    split_video(video, times, min_duration=min_dur)
