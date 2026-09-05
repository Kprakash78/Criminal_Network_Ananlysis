import os
import urllib.request
import time
from pathlib import Path

# We will just pull the necessary files manually and stick them in the HF cache manually
# to bypass the fragile huggingface_hub Windows symlink/timeout issues.

model_id = "google/flan-t5-large"
cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--google--flan-t5-large" / "snapshots" / "direct_download"
cache_dir.mkdir(parents=True, exist_ok=True)

files_to_download = {
    "config.json": "https://huggingface.co/google/flan-t5-large/resolve/main/config.json",
    "generation_config.json": "https://huggingface.co/google/flan-t5-large/resolve/main/generation_config.json",
    "special_tokens_map.json": "https://huggingface.co/google/flan-t5-large/resolve/main/special_tokens_map.json",
    "spiece.model": "https://huggingface.co/google/flan-t5-large/resolve/main/spiece.model",
    "tokenizer.json": "https://huggingface.co/google/flan-t5-large/resolve/main/tokenizer.json",
    "tokenizer_config.json": "https://huggingface.co/google/flan-t5-large/resolve/main/tokenizer_config.json",
    "model.safetensors": "https://huggingface.co/google/flan-t5-large/resolve/main/model.safetensors"
}

def download_file(url, filepath):
    tmp_path = filepath.with_suffix('.tmp')
    
    # Get total size
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=10) as r:
            total_size = int(r.headers.get('content-length', 0))
    except Exception as e:
        print(f"Failed to get HEAD for {url}: {e}")
        return False

    if filepath.exists() and filepath.stat().st_size == total_size:
        print(f"Already fully downloaded: {filepath.name}")
        return True

    downloaded = 0
    if tmp_path.exists():
        downloaded = tmp_path.stat().st_size

    if downloaded >= total_size:
        tmp_path.rename(filepath)
        return True

    print(f"Downloading {filepath.name}: {downloaded/1e6:.1f} MB / {total_size/1e6:.1f} MB")
    
    req = urllib.request.Request(url)
    req.add_header('Range', f'bytes={downloaded}-')
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            with open(tmp_path, 'ab') as f:
                while True:
                    chunk = response.read(1024 * 1024) # 1 MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (1024 * 1024 * 50) < (1024 * 1024): # print every ~50MB
                        print(f"  ... {downloaded/1e6:.1f} MB")
        
        tmp_path.rename(filepath)
        print(f"Finished {filepath.name}!")
        return True
    except Exception as e:
        print(f"Error during download of {filepath.name}: {e}")
        return False

success = True
for name, url in files_to_download.items():
    if not download_file(url, cache_dir / name):
        success = False
        break

if success:
    print("ALL FILES SUCCESSFULLY DOWNLOADED!")
    print(f"Model path: {cache_dir}")
