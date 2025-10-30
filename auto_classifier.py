# auto_image_classifier_optimized.py
import os
import sys
import math
import csv
import time
import shutil
import traceback
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

# progress
from tqdm import tqdm

# GUI
import tkinter as tk
from tkinter import filedialog, messagebox

# ML
try:
    import torch
except Exception:
    print("[ERROR] PyTorch is required. Install: pip install torch torchvision (choose correct CUDA/CPU wheel).")
    raise
try:
    import open_clip
except Exception:
    print("[ERROR] open-clip-torch is required. Install: pip install open-clip-torch")
    raise


# =========================
# Config (safe defaults)
# =========================
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
MODEL_NAME = "ViT-H-14"
MODEL_PRETRAINED = "laion2b_s32b_b79k"

# Start with a high-ish batch, will auto-decrease on OOM
INIT_BATCH_SIZE = 16

# Use FP16 autocast on CUDA/MPS for safe throughput & VRAM reduction
USE_AUTOCast = True

# Top-K mean per class (requested)
TOPK = 5

# Size balancing: adjusted = raw * sqrt(n_min / n_cls)
USE_CLASS_SIZE_WEIGHT = True

# Optional: minimum confidence (0..1) required to move; 0 means always move
MIN_CONFIDENCE_TO_MOVE = 0.0

# CSV log name
DEFAULT_LOG_NAME = f"classification_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"


# =========================
# Small utilities
# =========================
def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def scan_images_recursive(folder: Path) -> List[Path]:
    return [p for p in folder.rglob("*") if is_image_file(p)]

def safe_move(src: Path, dst_dir: Path) -> Path:
    """
    Move 'src' into 'dst_dir'. If name collision, append -1, -2, ...
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    stem, ext = src.stem, src.suffix
    cand = dst_dir / f"{stem}{ext}"
    k = 1
    while cand.exists():
        cand = dst_dir / f"{stem}-{k}{ext}"
        k += 1
    shutil.move(str(src), str(cand))
    return cand

def cos_to_conf(x: float) -> float:
    # Map cosine in [-1,1] -> [0,1]
    return float(max(0.0, min(1.0, (x + 1.0) / 2.0)))

def device_string() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def print_env():
    dev = device_string()
    print(f"[INFO] Device: {dev}")
    if dev == "cuda":
        print(f"[INFO] CUDA: {torch.version.cuda}")
        try:
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    print(f"[INFO] torch: {torch.__version__}")
    print(f"[INFO] open-clip: {open_clip.__version__}")


# =========================
# Embedder (RAM-only)
# =========================
class Embedder:
    def __init__(self, model_name=MODEL_NAME, pretrained=MODEL_PRETRAINED, use_autocast=USE_AUTOCast):
        self.device = torch.device(device_string())
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.use_autocast = use_autocast

        # detect autocast dtype + enabled flag
        self.autocast_kwargs = {}
        if self.use_autocast:
            if self.device.type == "cuda":
                self.autocast_kwargs = dict(enabled=True, dtype=torch.float16, device_type="cuda")
            elif self.device.type == "mps":
                # MPS autocast exists but dtype handling differs; keep it simple and safe
                self.autocast_kwargs = dict(enabled=True, device_type="mps")
            else:
                self.autocast_kwargs = dict(enabled=False)
        else:
            self.autocast_kwargs = dict(enabled=False)

        # set initial batch size (will reduce on OOM)
        self.batch_size = INIT_BATCH_SIZE

    @torch.no_grad()
    def embed_paths(self, paths: List[Path]) -> Tuple[np.ndarray, List[Path]]:
        """
        Embed a list of image paths -> (N, D) np.float32 normalized
        Skips unreadable files.
        """
        embs = []
        ok_paths = []

        i = 0
        N = len(paths)
        pbar = tqdm(total=N, desc="Embedding", unit="img")
        while i < N:
            bs = min(self.batch_size, N - i)
            batch_tensors = []
            batch_paths = []

            # load PIL -> preprocess -> tensor
            for j in range(bs):
                p = paths[i + j]
                try:
                    img = Image.open(p).convert("RGB")
                    t = self.preprocess(img)  # 3x224x224 float tensor
                    batch_tensors.append(t)
                    batch_paths.append(p)
                except UnidentifiedImageError:
                    print(f"[WARN] Unreadable image skipped: {p}")
                except Exception as e:
                    print(f"[WARN] Failed reading {p}: {e}")

            if not batch_tensors:
                i += bs
                pbar.update(bs)
                continue

            try:
                feats = self._forward_batch(batch_tensors)
                embs.append(feats)
                ok_paths.extend(batch_paths)
                i += bs
                pbar.update(bs)

            except RuntimeError as e:
                # Auto reduce batch size on OOM for max compatibility
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    if self.batch_size > 1:
                        self.batch_size = max(1, self.batch_size // 2)
                        print(f"[WARN] OOM detected. Reducing batch size to {self.batch_size} and retrying.")
                        torch.cuda.empty_cache() if self.device.type == "cuda" else None
                        continue
                    else:
                        print("[ERROR] OOM with batch size 1. Aborting this batch.")
                        i += bs
                        pbar.update(bs)
                        continue
                else:
                    print(f"[ERROR] Embed batch failed: {e}")
                    traceback.print_exc()
                    i += bs
                    pbar.update(bs)
                    continue

        pbar.close()

        if not embs:
            return np.empty((0, self.model.visual.output_dim), dtype=np.float32), []

        embs = np.vstack(embs).astype(np.float32)
        # L2 normalize
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        embs = embs / norms
        return embs, ok_paths

    def _forward_batch(self, batch_tensors: List[torch.Tensor]) -> np.ndarray:
        x = torch.stack(batch_tensors, dim=0).to(self.device, non_blocking=True)
        # AMP autocast for safe speed/VRAM
        with torch.autocast(**self.autocast_kwargs):
            feats = self.model.encode_image(x)
        feats = feats.detach().to("cpu").float().numpy()
        return feats


# =========================
# Scoring
# =========================
def class_scores_for_one(
    q: np.ndarray,
    class_embeds: Dict[str, np.ndarray],
    class_sizes: Dict[str, int],
    topk: int = TOPK,
    use_weight: bool = USE_CLASS_SIZE_WEIGHT
) -> Tuple[str, float, Dict[str, float]]:
    """
    q: (D,), already L2-normalized
    Returns:
      best_class (key of class_embeds), best_raw_score (float), scores_per_class (raw)
    Selection uses weighted scores, but we return best raw as confidence base.
    """
    if not class_embeds:
        return None, -1.0, {}

    scores_raw = {}
    scores_w = {}

    n_min = None
    if use_weight and class_sizes:
        n_min = max(1, min(class_sizes.values()))

    for cls, mat in class_embeds.items():
        if mat.size == 0:
            continue
        sims = mat @ q  # cosine since normalized
        k = min(topk, sims.size)
        if k <= 0:
            continue
        # top-k mean
        topk_vals = np.partition(sims, -k)[-k:]
        raw = float(topk_vals.mean())

        if use_weight and n_min is not None:
            n_cls = max(1, class_sizes.get(cls, mat.shape[0]))
            weight = math.sqrt(n_min / n_cls)
            w_score = raw * weight
        else:
            w_score = raw

        scores_raw[cls] = raw
        scores_w[cls] = w_score

    if not scores_w:
        return None, -1.0, {}

    best_cls = max(scores_w.items(), key=lambda kv: kv[1])[0]
    return best_cls, scores_raw[best_cls], scores_raw


# =========================
# Tk folder selection
# =========================
def ask_reference_dirs(max_dirs: int = 10) -> List[Path]:
    out = []
    cnt = 0
    while cnt < max_dirs:
        d = filedialog.askdirectory(title=f"Select reference (category) folder #{cnt+1}")
        if not d:
            break
        p = Path(d)
        if p in out:
            messagebox.showwarning("Duplicate", f"Already selected:\n{p}")
        else:
            out.append(p)
            cnt += 1
        if cnt < max_dirs:
            if not messagebox.askyesno("Continue?", "Select another reference folder?"):
                break
    return out

def ask_source_dirs() -> List[Path]:
    out = []
    while True:
        d = filedialog.askdirectory(title="Select a SOURCE folder (Cancel to finish)")
        if not d:
            break
        p = Path(d)
        if p not in out:
            out.append(p)
        if not messagebox.askyesno("Continue?", "Add another source folder?"):
            break
    return out


# =========================
# Main
# =========================
def main():
    print_env()

    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(
        "Image Auto-Classifier",
        "Select up to 10 reference (category) folders, then select one or more source folders to classify."
    )

    ref_dirs = ask_reference_dirs(max_dirs=10)
    if not ref_dirs:
        messagebox.showerror("No references", "No reference folders selected.")
        return

    src_dirs = ask_source_dirs()
    if not src_dirs:
        messagebox.showerror("No sources", "No source folders selected.")
        return

    # Build class -> list of reference images
    classes: Dict[str, List[Path]] = {}
    for d in ref_dirs:
        imgs = [p for p in d.iterdir() if is_image_file(p)]
        classes[str(d)] = imgs

    # Gather source images (recursive)
    src_imgs: List[Path] = []
    for d in src_dirs:
        src_imgs.extend(scan_images_recursive(d))

    # Exclude any image that lives inside reference dirs (do not reclassify exemplars)
    ref_set = set()
    for lst in classes.values():
        ref_set.update(map(str, lst))
    src_imgs = [p for p in src_imgs if str(p) not in ref_set]

    if not src_imgs:
        messagebox.showwarning("Nothing to classify", "No classifiable images found in source folders.")
        return

    # Load model
    print("[INFO] Loading OpenCLIP model & transforms...")
    embedder = Embedder(MODEL_NAME, MODEL_PRETRAINED)

    # Compute reference embeddings (RAM only)
    print("[INFO] Embedding reference images...")
    class_embeds: Dict[str, np.ndarray] = {}
    class_sizes: Dict[str, int] = {}
    for cls_dir_str, files in classes.items():
        files = [p for p in files if is_image_file(p)]
        if not files:
            class_embeds[cls_dir_str] = np.empty((0, 1024), dtype=np.float32)
            class_sizes[cls_dir_str] = 0
            print(f"[WARN] Empty reference folder: {cls_dir_str}")
            continue

        embs, ok_paths = embedder.embed_paths(files)
        class_embeds[cls_dir_str] = embs
        class_sizes[cls_dir_str] = embs.shape[0]
        print(f"[INFO] Ref '{Path(cls_dir_str).name}': {embs.shape[0]} embeddings")

    # Ensure at least one class has data
    if all(arr.size == 0 for arr in class_embeds.values()):
        messagebox.showerror("No usable references", "All reference folders are empty or unreadable.")
        return

    # Classify sources
    log_rows = []
    print(f"[INFO] Classifying {len(src_imgs)} images...")
    for p in tqdm(src_imgs, desc="Classifying", unit="img"):
        try:
            embs, ok = embedder.embed_paths([p])
            if embs.shape[0] == 0:
                print(f"[WARN] Skip unreadable: {p}")
                continue
            q = embs[0]  # (D,)

            best_cls, best_raw, _ = class_scores_for_one(
                q, class_embeds, class_sizes, topk=TOPK, use_weight=USE_CLASS_SIZE_WEIGHT
            )
            if best_cls is None:
                print(f"[WARN] No class scored: {p}")
                continue

            conf = cos_to_conf(best_raw)
            if conf < MIN_CONFIDENCE_TO_MOVE:
                # If you want to skip low-confidence moves, handle here.
                # For now, proceed as requested.
                pass

            dst_dir = Path(best_cls)
            new_path = safe_move(p, dst_dir)

            log_rows.append([str(p), str(new_path), f"{conf:.4f}"])

        except Exception as e:
            print(f"[ERROR] Failed on {p}: {e}")
            traceback.print_exc()

    # Save CSV log
    if log_rows:
        log_path = Path.cwd() / DEFAULT_LOG_NAME
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["original_path", "new_path", "confidence"])
            w.writerows(log_rows)
        print(f"[INFO] Done. Log saved: {log_path}")
        messagebox.showinfo("Done", f"Classified {len(log_rows)} images.\nLog: {log_path}")
    else:
        print("[INFO] No images were moved.")
        messagebox.showwarning("No moves", "No images were moved / classified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc()
        sys.exit(1)
