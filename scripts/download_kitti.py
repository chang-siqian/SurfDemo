"""Multi-threaded KITTI scene flow dataset downloader."""
import os
import sys
import threading
import time
from urllib.request import Request, urlopen

URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip"
OUTPUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "KITTI", "data_scene_flow.zip")
NUM_THREADS = 8
CHUNK_SIZE = 1024 * 1024  # 1MB per chunk

def get_file_size(url):
    req = Request(url, method="HEAD")
    resp = urlopen(req, timeout=30)
    size = int(resp.headers.get("Content-Length", 0))
    resp.close()
    return size

def download_range(url, start, end, output, idx):
    headers = {"Range": f"bytes={start}-{end}"}
    req = Request(url, headers=headers)
    for attempt in range(3):
        try:
            resp = urlopen(req, timeout=60)
            written = 0
            with open(output, "r+b") as f:
                f.seek(start)
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    written += len(chunk)
            resp.close()
            return written
        except Exception as e:
            print(f"  [Thread {idx}] attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return 0

def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    print(f"Getting file size from {URL}...")
    total_size = get_file_size(URL)
    print(f"Total size: {total_size / (1024**2):.1f} MB ({total_size} bytes)")

    # Pre-allocate file
    if not os.path.exists(OUTPUT):
        with open(OUTPUT, "wb") as f:
            f.truncate(total_size)
        print("File pre-allocated.")
    else:
        print("File already exists, will overwrite.")

    part_size = total_size // NUM_THREADS
    threads = []
    results = [0] * NUM_THREADS

    start_time = time.time()

    for i in range(NUM_THREADS):
        start = i * part_size
        end = start + part_size - 1 if i < NUM_THREADS - 1 else total_size - 1
        t = threading.Thread(
            target=lambda idx=i, s=start, e=end: results.__setitem__(idx, download_range(URL, s, e, OUTPUT, idx)),
            daemon=True
        )
        threads.append(t)
        t.start()
        print(f"  Started thread {i}: bytes {start}-{end}")

    print(f"\nDownloading with {NUM_THREADS} threads...")

    # Progress display
    while any(t.is_alive() for t in threads):
        if os.path.exists(OUTPUT):
            current = os.path.getsize(OUTPUT)
            pct = current * 100.0 / total_size
            elapsed = time.time() - start_time
            speed = current / (elapsed * 1024 * 1024) if elapsed > 0 else 0
            print(f"\r  {current / (1024**2):.0f} / {total_size / (1024**2):.0f} MB ({pct:.1f}%) - {speed:.1f} MB/s   ", end="")
        time.sleep(1)

    # Final check
    for t in threads:
        t.join()

    total_written = sum(results)
    elapsed = time.time() - start_time
    final_size = os.path.getsize(OUTPUT)
    print(f"\rDone! {final_size / (1024**2):.0f} MB in {elapsed:.1f}s ({final_size/elapsed/(1024**2):.1f} MB/s)")

    if final_size < total_size:
        print(f"WARNING: only downloaded {final_size}/{total_size} bytes!")
        return 1

    print("Download complete. Verifying zip integrity...")
    import zipfile
    try:
        with zipfile.ZipFile(OUTPUT, 'r') as zf:
            names = zf.namelist()
            print(f"Zip OK. {len(names)} files inside.")
            for name in names[:10]:
                print(f"  {name}")
            if len(names) > 10:
                print(f"  ... and {len(names)-10} more")
    except zipfile.BadZipFile:
        print("ERROR: Downloaded file is not a valid zip!")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
