import os
import io
import re
import sys
import math
import base64
import posixpath
import hashlib
import tempfile
import concurrent.futures
import threading
from pathlib import Path
from typing import List, Tuple, Iterable, Dict

import yaml
import numpy as np
from tqdm import tqdm

from adbutils import adb

from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import open_clip

global_lock = threading.Lock()


# ---------------------------- Config ----------------------------

EMULATED0 = "/storage/emulated/0"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif", ".heic", ".avif"}

def urlsafe_b64_without_padding(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")

def b64_path(rel_path: str) -> str:
    # rel_path begins with leading slash, e.g. "/a/b.jpg"
    return urlsafe_b64_without_padding(rel_path.encode("utf-8"))

def sha256_file(path: Path) -> bytes:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.digest()

def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # defaults
    cfg.setdefault("device_serial", None)
    cfg.setdefault("adb_find_cmd", "find")
    cfg.setdefault("target_folders", [])
    cfg.setdefault("ignore_folder_regex", [])
    cfg.setdefault("local_root", ".")
    cfg.setdefault("adb_cache_dir", "_adb_cache")
    cfg.setdefault("emb_cache_dir", "_emb_cache")
    cfg.setdefault("csv_path", "index.csv")
    cfg.setdefault("pull_threads", 8)
    cfg.setdefault("batch_size", 64)
    cfg.setdefault("num_workers", 4)
    cfg.setdefault("use_fp16", True)
    cfg.setdefault("device", "auto")
    cfg.setdefault("max_image_pixels", None)
    cfg.setdefault("jpeg_quality", 92)
    return cfg


# ---------------------------- ADB listing ----------------------------

def list_files_under(dev, abs_dir: str, find_cmd: str = "find") -> List[str]:
    # Use POSIX paths on device
    cmd = f"{find_cmd} {abs_dir} -type f 2>/dev/null"
    with global_lock:
        out = dev.shell(cmd)
    files = [line.strip() for line in out.splitlines() if line.strip()]
    return files

def filter_targets(files: Iterable[str],
                   ignore_res: List[re.Pattern]) -> List[str]:
    kept = []
    for f in files:
        if not f.startswith(EMULATED0 + "/") and f != EMULATED0:
            # not under /storage/emulated/0; skip
            continue
        rel = f[len(EMULATED0):] or "/"
        # keep only image-like by extension, case-insensitive
        ext = os.path.splitext(f)[1].lower()
        if ext not in IMG_EXTS:
            continue
        # check ignored folder regex on the FOLDER part of the relative path
        folder_rel = posixpath.dirname(rel)
        ignore = False
        for rx in ignore_res:
            if rx.search(folder_rel):
                ignore = True
                break
        if not ignore:
            kept.append(f)
    return kept


# ---------------------------- Pull + resize workers ----------------------------

def pil_safe_open_resize_224(src_path: Path, dst_path: Path, jpeg_quality: int, max_pixels: int | None):
    # Open and resize to 224x224. Handle large images and EXIF orientation.
    if max_pixels is None:
        Image.MAX_IMAGE_PIXELS = None
    else:
        Image.MAX_IMAGE_PIXELS = max_pixels

    with Image.open(src_path) as im:
        im = ImageOps.exif_transpose(im)
        # Remove alpha, enforce RGB
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        elif im.mode == "L":
            im = im.convert("RGB")
        im = im.resize((224, 224), resample=Image.BICUBIC)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        ext = dst_path.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            im.save(dst_path, format="JPEG", quality=jpeg_quality, optimize=True)
        elif ext == ".png":
            im.save(dst_path, format="PNG", optimize=True)
        else:
            # fallback to PNG for non-lossy formats if extension unsupported by PIL save
            im.save(dst_path, format="PNG", optimize=True)

def pull_and_resize_worker(dev, remote_path: str, emu_root: str,
                           adb_cache_dir: Path, jpeg_quality: int, max_pixels: int | None) -> Tuple[str, str]:
    """
    Returns tuple: (relative_path, local_cached_filename)
      relative_path: like "/a.jpg"
      local_cached_filename: like "QmFzZTY0...=.jpg" (padding stripped)
    """
    rel = remote_path[len(emu_root):]
    if not rel.startswith("/"):
        rel = "/" + rel
    ext = os.path.splitext(remote_path)[1].lower()
    # Build cache filename
    digest = hashlib.sha256(rel.encode("utf-8")).digest()
    safe_base = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")[:64]
    out_name = safe_base + ext
    out_path = adb_cache_dir / out_name

    if out_path.exists():
        return rel, out_name

    # pull to temp and resize
    tmp_pull = out_path.with_suffix(out_path.suffix + ".part_pull")
    tmp_pull.parent.mkdir(parents=True, exist_ok=True)
    try:
        dev.sync.pull(remote_path, str(tmp_pull))
    except Exception as e:
        raise RuntimeError(f"ADB pull failed: {remote_path} ({e})")

    tmp_resized = out_path.with_suffix(out_path.suffix + ".part_resize")
    try:
        pil_safe_open_resize_224(tmp_pull, tmp_resized, jpeg_quality, max_pixels)
        os.replace(tmp_resized, out_path)
    finally:
        try:
            tmp_resized.exists() and tmp_resized.unlink()
        except Exception:
            pass
        try:
            tmp_pull.exists() and tmp_pull.unlink()
        except Exception:
            pass

    return rel, out_name


# ---------------------------- Embedding ----------------------------

class CachedImageDataset(Dataset):
    def __init__(self, files: List[Path], normalize: T.Normalize):
        self.files = files
        self.t = T.Compose([
            T.ToTensor(),
            normalize
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        with Image.open(p) as im:
            im = im.convert("RGB")
            x = self.t(im)
        return x, p.name  # return filename to map outputs

def build_openclip(device: torch.device):
    model_name = "ViT-H-14"
    pretrained = "laion2b_s32b_b79k"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    # Use only normalization from preprocess, since we already resized to 224x224.
    # open_clip preprocess is a torchvision transform pipeline; extract its Normalize.
    normalize = None
    for t in getattr(preprocess, "transforms", []):
        if isinstance(t, T.Normalize):
            normalize = t
            break
    if normalize is None:
        # Fallback to CLIP mean/std
        normalize = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=(0.26862954, 0.26130258, 0.27577711))
    model.eval()
    return model, normalize

def compute_embeddings(adb_cache_dir: Path, emb_cache_dir: Path,
                       batch_size: int, num_workers: int, use_fp16: bool, device_pref: str) -> Dict[str, str]:
    """
    Returns mapping: cached_image_filename -> emb_filename
    Only computes for missing .npy files.
    """
    emb_cache_dir.mkdir(parents=True, exist_ok=True)

    # Collect to-run and also prepare mapping of desired output names
    to_process: List[Path] = []
    desired_names: Dict[str, str] = {}

    for img_path in adb_cache_dir.iterdir():
        if not img_path.is_file():
            continue
        ext = img_path.suffix.lower()
        if ext not in IMG_EXTS and ext not in {".jpeg"}:
            continue
        # hash of file bytes
        digest = sha256_file(img_path)
        emb_base64 = urlsafe_b64_without_padding(digest)[:64]
        emb_name = emb_base64 + ".npy"
        desired_names[img_path.name] = emb_name
        if not (emb_cache_dir / emb_name).exists():
            to_process.append(img_path)

    if not to_process:
        # nothing to do
        return desired_names

    # Device select
    if device_pref == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_pref)
    fp16 = use_fp16 and device.type == "cuda"

    model, normalize = build_openclip(device)
    dataset = CachedImageDataset(to_process, normalize)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))

    dtype = torch.float16 if fp16 else torch.float32

    with torch.no_grad():
        pbar = tqdm(total=len(dataset), desc="Embedding", unit="img")
        for batch, names in loader:
            batch = batch.to(device=device, dtype=dtype, non_blocking=True)
            feats = model.encode_image(batch)
            feats = feats.float()  # save as float32 for portability
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            feats_np = feats.cpu().numpy()
            for i, filename in enumerate(names):
                emb_name = desired_names[filename]
                out_path = emb_cache_dir / emb_name
                np.save(out_path, feats_np[i])
            pbar.update(len(names))
        pbar.close()

    return desired_names


# ---------------------------- Main pipeline ----------------------------

def main(cfg_path: str = "config.yml"):
    cfg = load_config(cfg_path)

    # PIL safety
    if cfg["max_image_pixels"] is None:
        Image.MAX_IMAGE_PIXELS = None
    else:
        Image.MAX_IMAGE_PIXELS = int(cfg["max_image_pixels"])

    local_root = Path(cfg["local_root"]).resolve()
    adb_cache_dir = (local_root / cfg["adb_cache_dir"]).resolve()
    emb_cache_dir = (local_root / cfg["emb_cache_dir"]).resolve()
    csv_path = (local_root / cfg["csv_path"]).resolve()

    adb_cache_dir.mkdir(parents=True, exist_ok=True)
    emb_cache_dir.mkdir(parents=True, exist_ok=True)

    # Connect device
    dev = adb.device(serial=cfg["device_serial"])

    # Compile ignore regex
    ignore_res = [re.compile(p) for p in cfg["ignore_folder_regex"]]

    # Collect remote files from targets
    all_remote_files: List[str] = []
    targets = cfg["target_folders"]
    if not targets:
        print("No target_folders configured.", file=sys.stderr)
        sys.exit(1)

    for rel in targets:
        if not rel.startswith("/"):
            rel = "/" + rel
        abs_dir = posixpath.normpath(EMULATED0 + rel)
        files = list_files_under(dev, abs_dir, cfg["adb_find_cmd"])
        all_remote_files.extend(files)

    # Filter by ignore rules and image extensions
    remote_images = filter_targets(all_remote_files, ignore_res)

    print(f"Found {len(remote_images)} candidate images under targets.")

    # Pull + resize multithreaded
    pull_threads = max(1, int(cfg["pull_threads"]))

    pulled_records: List[Tuple[str, str]] = []  # (rel_path, adb_cache_filename)
    errors = 0

    dev_pool = [adb.device(serial=cfg["device_serial"]) for _ in range(pull_threads * 4)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=pull_threads) as ex:
        futures = []
        for i, rp in enumerate(remote_images):
            futures.append(ex.submit(
                pull_and_resize_worker,
                dev_pool[i % (pull_threads * 4)], rp, EMULATED0, adb_cache_dir,
                int(cfg["jpeg_quality"]), cfg["max_image_pixels"]
            ))
        for fut in tqdm(concurrent.futures.as_completed(futures),
                        total=len(futures),
                        desc="Pull+Resize", unit="file"):
            try:
                rel, name = fut.result()
                pulled_records.append((rel, name))
            except Exception as e:
                errors += 1
                print(f"[WARN] {e}", file=sys.stderr)

    print(f"Pull+Resize done. {len(pulled_records)} ok, {errors} failed.")

    # Compute embeddings
    mapping_img_to_emb = compute_embeddings(
        adb_cache_dir=adb_cache_dir,
        emb_cache_dir=emb_cache_dir,
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        use_fp16=bool(cfg["use_fp16"]),
        device_pref=str(cfg["device"])
    )

    # Emit CSV
    # Columns: relative_path, adb_cache_filename, emb_cache_filename
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relative_path", "adb_cache_filename", "emb_cache_filename"])
        for rel, adb_name in pulled_records:
            emb_name = mapping_img_to_emb.get(adb_name, "")
            w.writerow([rel, adb_name, emb_name])

    print(f"Wrote CSV: {csv_path}")
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("config.yml")
