import os
import sys
import cv2
import time
import threading
import concurrent.futures
from pathlib import Path
from adbutils import adb

# ---------------- Config ----------------
REMOTE_DIR = "/storage/emulated/0"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
MAX_WORKERS = 8
POOL_MULTIPLIER = 4   # number of adb instances per thread
TMP_DIR = "_tmp_check"
os.makedirs(TMP_DIR, exist_ok=True)


# ---------------- Device Pool ----------------
class DevicePool:
    """Thread-safe pool with multiple adb instances for one phone"""
    def __init__(self, multiplier=4):
        devs = adb.device_list()
        if not devs:
            raise RuntimeError("No adb devices connected")
        base = devs[0]
        self.devices = [adb.device(serial=base.serial) for _ in range(MAX_WORKERS * multiplier)]
        self.lock = threading.Lock()
        self.index = 0

    def acquire(self):
        dev = self.devices[self.index]
        self.index = (self.index + 1) % len(self.devices)
        return dev


# ---------------- Utility ----------------
def list_remote_images(dev, remote_dir):
    """List all images except hidden and /Android"""
    cmd = (
        f"find {remote_dir} -type f "
        "\\( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \\) 2>/dev/null"
    )
    result = dev.shell(cmd).splitlines()
    clean = []
    for r in result:
        r = r.strip()
        if not r:
            continue
        # skip hidden or Android dirs
        if "/Android" in r or "/." in r or r.split("/")[-1].startswith("."):
            continue
        clean.append(r)
    print(f"[INFO] Found {len(clean)} images to check.")
    return clean


def is_remote_image_broken(dev, path):
    """Pull remote file temporarily and check via OpenCV"""
    tmp_path = Path(TMP_DIR) / (str(abs(hash(path))) + ".jpg")
    try:
        dev.sync.pull(path, str(tmp_path))
        img = cv2.imread(str(tmp_path))
        ok = img is not None and img.size > 0
    except Exception:
        ok = False
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return not ok


def worker(pool, path, results):
    dev = pool.acquire()
    broken = is_remote_image_broken(dev, path)
    if broken:
        results.append(path)
        print(f"[BROKEN] {path}")
    return broken


# ---------------- Main ----------------
def main():
    pool = DevicePool(multiplier=POOL_MULTIPLIER)
    print(f"[INFO] Using {len(pool.devices)} adb instances ({MAX_WORKERS} threads).")
    dev0 = pool.acquire()
    paths = list_remote_images(dev0, REMOTE_DIR)

    broken = []
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = [exe.submit(worker, pool, p, broken) for p in paths]
        for i, _ in enumerate(concurrent.futures.as_completed(futures), 1):
            sys.stdout.write(f"\rChecked {i}/{len(paths)}")
            sys.stdout.flush()
    print()

    print(f"[INFO] Total broken files: {len(broken)} (checked {len(paths)} in {time.time() - start:.1f}s)")
    if broken:
        print("\nBroken file list:")
        for p in broken:
            print(p)
        confirm = input("\nDelete all these files? (y/N): ").strip().lower()
        if confirm == "y":
            for p in broken:
                dev0.shell(f"rm -f '{p}'")
            print("Deleted.")
        else:
            print("Aborted.")
    else:
        print("No broken files found.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
