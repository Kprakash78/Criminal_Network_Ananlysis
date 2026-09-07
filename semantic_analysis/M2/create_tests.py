import os
import subprocess
# pyrefly: ignore [missing-import]
import imageio_ffmpeg
from urllib.request import urlretrieve

def create_test_videos():
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    print("Downloading preamble speech audio...")
    audio_path = "preamble.aiff"
    if not os.path.exists(audio_path):
        try:
            urlretrieve("https://www2.cs.uic.edu/~i101/SoundFiles/preamble.wav", audio_path)
            print("Downloaded successfully.")
        except Exception as e:
            print("Failed to download fallback to mac say if available...")
            subprocess.run(["say", "We the people are forming a perfect union", "-o", audio_path])
    
    print("Synthesizing test_speech.mp4 (has audio)...")
    subprocess.run([
        ffmpeg_exe, "-y", "-f", "lavfi", "-i", "color=c=black:s=640x360:d=10",
        "-i", audio_path, "-c:v", "libx264", "-c:a", "aac", "-shortest", "test_speech.mp4"
    ], check=True)
    
    print("Synthesizing test_silent.mp4 (pure silence)...")
    subprocess.run([
        ffmpeg_exe, "-y", "-f", "lavfi", "-i", "color=c=black:s=640x360:d=5",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", "test_silent.mp4"
    ], check=True)

if __name__ == '__main__':
    create_test_videos()
    print("Dummy videos generation completed.")
