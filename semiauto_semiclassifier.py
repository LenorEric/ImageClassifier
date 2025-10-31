import os
import io
import cv2
import sys
import time
import math
import yaml
import ftplib
import random
import shutil
import socket
import hashlib
import zipfile
import tempfile
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import numpy as np
from PIL import Image

# --- OpenCLIP ---
try:
    import open_clip
except Exception as e:
    print("Failed to import open_clip. Install with: pip install open_clip_torch", file=sys.stderr)
    raise

# -------------------------
# Config dataclasses
# -------------------------
@dataclass
class FTPConfig:
    host: str
    port: int = 21
    root_dir: str = "/"
    passive_mode: bool = True

@dataclass
class CacheConfig:
    dir: str = "./_ftp_cache"

@dataclass
class ImagesConfig:
    exts: List[str] = None
    recursive: bool = True
    max_per_dir: int = 0

@dataclass
class EmbeddingConfig:
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 0
    topk: int = 10

@dataclass
class ReviewConfig:
    seed: int = 42
    sample_per_band: int = 10  # total 50 across 5 bands

@dataclass
class DisplayConfig:
    window_name: str = "Similarity Review"
    default_window_wh: Tuple[int, int] = (900, 1600)  # (H, W)
    font_scale: float = 0.6
    font_thick: int = 1

@dataclass
class LogConfig:
    csv_path: str = "./move_decisions.csv"

@dataclass
class LimitsConfig:
    max_others_to_embed: int = 0

# -------------------------
# Utility functions
# -------------------------
def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Defaults and normalization
    ftp_cfg = FTPConfig(**cfg.get("ftp", {}))
    cache_cfg = CacheConfig(**cfg.get("cache", {}))
    images_cfg = ImagesConfig(**cfg.get("images", {}))
    if not images_cfg.exts:
        images_cfg.exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    images_cfg.exts = [e.lower() for e in images_cfg.exts]

    emb_cfg = EmbeddingConfig(**cfg.get("embedding", {}))
    review_cfg = ReviewConfig(**cfg.get("review", {}))
    display_cfg = DisplayConfig(**cfg.get("display", {}))
    log_cfg = LogConfig(**cfg.get("log", {}))
    limits_cfg = LimitsConfig(**cfg.get("limits", {}))
    return ftp_cfg, cache_cfg, images_cfg, emb_cfg, review_cfg, display_cfg, log_cfg, limits_cfg

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def now_ts():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def hash_path(remote_path: str) -> str:
    # stable hashed filename prefix to avoid collisions
    h = hashlib.sha1(remote_path.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return h

def is_image_name(name: str, exts: List[str]) -> bool:
    return os.path.splitext(name)[1].lower() in exts

# -------------------------
# FTP helpers
# -------------------------
class FTPClient:
    def __init__(self, cfg: FTPConfig):
        self.cfg = cfg
        self.ftp = ftplib.FTP()
        self.ftp.connect(cfg.host, cfg.port, timeout=30)
        # Anonymous login
        self.ftp.login()  # by default anonymous
        self.ftp.set_pasv(cfg.passive_mode)
        self.cwd(cfg.root_dir)

    def cwd(self, path: str):
        self.ftp.cwd(path)

    def pwd(self) -> str:
        return self.ftp.pwd()

    def list_dir(self, path: str) -> List[Tuple[str, bool]]:
        """
        Returns list of (name, is_dir) within path.
        Tries MLSD, falls back to NLST + CWD probes.
        """
        items: List[Tuple[str, bool]] = []
        try:
            # MLSD provides facts including type
            self.ftp.cwd(path)
            for name, facts in self.ftp.mlsd():
                # skip self/parent
                if name in (".", ".."):
                    continue
                is_dir = facts.get("type", "") == "dir"
                items.append((name, is_dir))
        except Exception:
            # Fallback
            self.ftp.cwd(path)
            names = self.ftp.nlst()
            for name in names:
                if name in (".", ".."):
                    continue
                is_dir = False
                try:
                    self.ftp.cwd(os.path.join(path, name))
                    is_dir = True
                    self.ftp.cwd(path)  # back
                except Exception:
                    is_dir = False
                items.append((os.path.basename(name), is_dir))
        return items

    def walk_images(self, root: str, exts: List[str], recursive: bool, max_per_dir: int = 0) -> List[str]:
        """
        Returns list of absolute remote file paths under root that are images.
        """
        stack = [root]
        files = []
        while stack:
            d = stack.pop()
            try:
                entries = self.list_dir(d)
            except Exception as e:
                print(f"[WARN] list_dir failed at {d}: {e}", file=sys.stderr)
                continue
            count_in_dir = 0
            for name, is_dir in entries:
                rp = os.path.join(d, name).replace("\\", "/")
                if is_dir:
                    if recursive:
                        stack.append(rp)
                else:
                    if is_image_name(name, exts):
                        files.append(rp)
                        count_in_dir += 1
                        if max_per_dir and count_in_dir >= max_per_dir:
                            # stop collecting this dir if capped
                            pass
        return sorted(files)

    def download_file(self, remote_path: str, local_path: str):
        # ensure local dir
        ensure_dir(os.path.dirname(local_path))
        with open(local_path, "wb") as f:
            self.ftp.retrbinary(f"RETR {remote_path}", f.write)

    def file_exists(self, remote_path: str) -> bool:
        # Try to get size or list parent
        parent = os.path.dirname(remote_path).replace("\\", "/") or "/"
        name = os.path.basename(remote_path)
        try:
            for n, is_dir in self.list_dir(parent):
                if n == name and not is_dir:
                    return True
        except Exception:
            pass
        return False

    def dir_exists(self, remote_dir: str) -> bool:
        parent = os.path.dirname(remote_dir).replace("\\", "/") or "/"
        name = os.path.basename(remote_dir.rstrip("/"))
        try:
            for n, is_dir in self.list_dir(parent):
                if n == name and is_dir:
                    return True
        except Exception:
            pass
        return False

    def move_file(self, src: str, dst_dir: str) -> str:
        """
        Move src into dst_dir. Auto-rename to avoid collisions.
        Returns new remote path.
        """
        if not self.dir_exists(dst_dir):
            raise RuntimeError(f"Target directory does not exist on FTP: {dst_dir}")
        base = os.path.basename(src)
        name, ext = os.path.splitext(base)
        candidate = os.path.join(dst_dir, base).replace("\\", "/")

        # if no conflict, rename directly
        if not self.file_exists(candidate):
            self._rename(src, candidate)
            return candidate

        # else try suffixes
        for i in range(1, 10000):
            base2 = f"{name}-{i}{ext}"
            candidate2 = os.path.join(dst_dir, base2).replace("\\", "/")
            if not self.file_exists(candidate2):
                self._rename(src, candidate2)
                return candidate2
        raise RuntimeError("Auto-rename exceeded 9999 attempts")

    def _rename(self, src: str, dst: str):
        # Ensure dst parent exists and is accessible
        self.ftp.rename(src, dst)

    def close(self):
        try:
            self.ftp.quit()
        except Exception:
            try:
                self.ftp.close()
            except Exception:
                pass

# -------------------------
# OpenCLIP model loader
# -------------------------
def load_openclip(model_name: str, pretrained: str, device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        device=device
    )
    model.eval()
    return model, preprocess

def image_to_tensor(img_path: str, preprocess):
    img = Image.open(img_path).convert("RGB")
    return preprocess(img)

def batched_embed(image_paths: List[str], preprocess, model, device: str, batch_size: int) -> np.ndarray:
    tensors = []
    embs = []
    total = len(image_paths)
    for i, p in enumerate(image_paths):
        try:
            tensors.append(image_to_tensor(p, preprocess))
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}", file=sys.stderr)
            tensors.append(None)

        # flush batches
        if len(tensors) == batch_size or i == total - 1:
            batch = [t for t in tensors if t is not None]
            idx_mask = [t is not None for t in tensors]
            if batch:
                with torch.no_grad():
                    x = torch.stack(batch, dim=0).to(device)
                    feats = model.encode_image(x)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    feats = feats.detach().cpu().numpy()
                # scatter back into the same order using idx_mask
                it = iter(feats)
                for ok in idx_mask:
                    if ok:
                        embs.append(next(it))
                    else:
                        embs.append(None)
            else:
                embs += [None] * len(tensors)
            tensors = []

    # consolidate into ndarray, dropping Nones and keeping order with None entries as needed
    # For downstream we will filter out Nones and sync lists accordingly
    return np.array([e if e is not None else np.zeros((model.visual.output_dim,), dtype=np.float32) for e in embs], dtype=np.float32)

# -------------------------
# Similarity scoring
# -------------------------
def avg_topk_sim(other_embs: np.ndarray, target_embs: np.ndarray, topk: int) -> np.ndarray:
    """
    other_embs: [M, D] normalized
    target_embs: [T, D] normalized
    returns scores [M]
    """
    if len(target_embs) == 0 or len(other_embs) == 0:
        return np.zeros((len(other_embs),), dtype=np.float32)
    # torch for speed
    with torch.no_grad():
        t = torch.from_numpy(target_embs)  # [T, D]
        o = torch.from_numpy(other_embs)   # [M, D]
        sims = o @ t.T                     # [M, T]
        k = min(topk, t.shape[0])
        topk_vals = torch.topk(sims, k=k, dim=1).values  # [M, k]
        mean_vals = topk_vals.mean(dim=1)                # [M]
    return mean_vals.numpy()

# -------------------------
# Weighted threshold from labeled sample
# -------------------------
def weighted_threshold(similarities: List[float], decisions: List[int]) -> float:
    """
    similarities: list of similarity scores
    decisions: 1 if moved (positive), 0 if ignored (negative)
    Returns threshold t minimizing weighted 0/1 loss with class weights:
      w_pos = 0.5 / n_pos; w_neg = 0.5 / n_neg
    """
    x = np.array(similarities, dtype=np.float64)
    y = np.array(decisions, dtype=np.int32)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        # degenerate; pick midpoint
        return float((x.min() + x.max()) / 2.0)
    w_pos = 0.5 / n_pos
    w_neg = 0.5 / n_neg

    # Candidate thresholds between sorted unique scores, plus -inf/+inf guards
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]

    # Precompute cumulative weighted positives/negatives
    wpos = (ys == 1).astype(float) * w_pos
    wneg = (ys == 0).astype(float) * w_neg
    cum_pos = np.cumsum(wpos)            # weight of positives up to i
    cum_neg = np.cumsum(wneg)            # weight of negatives up to i
    total_pos = cum_pos[-1]
    total_neg = cum_neg[-1]

    # If threshold t predicts y=1 when x >= t:
    # error(t) = weight(pos with x<t) + weight(neg with x>=t)
    # Evaluate at cuts between xs[i] and xs[i+1]
    best_err = float("inf")
    best_t = float(xs.mean())
    # Include cut before first and after last
    cuts = [-float("inf")] + [ (xs[i] + xs[i+1]) / 2.0 for i in range(len(xs)-1) ] + [float("inf")]

    for i, t in enumerate(cuts):
        # index r = last index where xs[r] < t
        r = np.searchsorted(xs, t, side="left") - 1
        if r < 0:
            pos_left = 0.0
            neg_left = 0.0
        else:
            pos_left = cum_pos[r]
            neg_left = cum_neg[r]
        # neg with x >= t has weight total_neg - neg_left
        err = pos_left + (total_neg - neg_left)
        if err < best_err:
            best_err = err
            best_t = t
    # Clamp finite
    if not np.isfinite(best_t):
        best_t = float((x.min() + x.max()) / 2.0)
    return float(best_t)

# -------------------------
# Display
# -------------------------
def get_window_size(win_name: str, fallback_hw: Tuple[int, int]) -> Tuple[int, int]:
    try:
        x, y, w, h = cv2.getWindowImageRect(win_name)
        if w > 0 and h > 0:
            return h, w
    except Exception:
        pass
    return fallback_hw

def scale_to_fit(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def put_text_top(img: np.ndarray, text: str, font_scale: float, thick: int):
    out = img.copy()
    y0 = 24
    cv2.putText(out, text, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thick, cv2.LINE_AA)
    return out

# -------------------------
# Main flow
# -------------------------
def pick_target_dir_interactive(ftp: FTPClient, root: str) -> str:
    # simple console navigation within root
    cur = root
    while True:
        items = ftp.list_dir(cur)
        dirs = [(name, True) for name, isdir in items if isdir]
        files = [(name, False) for name, isdir in items if not isdir]
        print(f"\n[DIR] {cur}")
        for i, (name, _) in enumerate(dirs):
            print(f"  [{i}] {name}/")
        print("  [s] select current as TARGET")
        print("  [u] up one level")
        choice = input("Choose dir index to enter, or 's' to select, 'u' to go up: ").strip()
        if choice.lower() == 's':
            return cur
        if choice.lower() == 'u':
            if cur.rstrip("/") != "/":
                cur = os.path.dirname(cur.rstrip("/"))
                if not cur:
                    cur = "/"
            else:
                print("Already at root.")
            continue
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(dirs):
                cur = os.path.join(cur, dirs[idx][0]).replace("\\", "/")
            else:
                print("Invalid index.")
        else:
            print("Invalid input.")

def build_local_cache_path(cache_dir: str, remote_path: str) -> str:
    base = os.path.basename(remote_path)
    h = hash_path(remote_path)
    # Keep extension
    name, ext = os.path.splitext(base)
    subdir = os.path.dirname(remote_path).strip("/").replace("/", "_")[:64]
    return os.path.join(cache_dir, subdir, f"{name}.{h}{ext}")

def download_many(ftp: FTPClient, remote_paths: List[str], cache_dir: str) -> List[str]:
    locals_ = []
    for rp in remote_paths:
        lp = build_local_cache_path(cache_dir, rp)
        if not os.path.exists(lp):
            try:
                ftp.download_file(rp, lp)
            except Exception as e:
                print(f"[WARN] Download failed {rp}: {e}", file=sys.stderr)
                lp = ""  # mark failed
        locals_.append(lp)
    return locals_

def log_csv_append(csv_path: str, rows: List[Tuple[str, str, float, str, str]]):
    # columns: original_path,new_path,similarity,decision,stage,timestamp
    header_needed = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("original_path,new_path,similarity,decision,stage,timestamp\n")
        for (orig, newp, sim, decision, stage) in rows:
            f.write(f"{orig},{newp},{sim:.6f},{decision},{stage},{now_ts()}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python ftp_similarity_reviewer.py config.yml")
        sys.exit(1)

    ftp_cfg, cache_cfg, images_cfg, emb_cfg, review_cfg, display_cfg, log_cfg, limits_cfg = load_config(sys.argv[1])
    ensure_dir(cache_cfg.dir)

    # Connect FTP
    print(f"Connecting FTP {ftp_cfg.host}:{ftp_cfg.port} ...")
    try:
        ftp = FTPClient(ftp_cfg)
    except Exception as e:
        print(f"[ERROR] FTP connect/login failed: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        root = ftp_cfg.root_dir
        print(f"Root: {root}")
        target_dir = pick_target_dir_interactive(ftp, root)
        print(f"Selected TARGET: {target_dir}")

        # List all images
        print("Listing target images ...")
        target_imgs = ftp.walk_images(target_dir, images_cfg.exts, images_cfg.recursive, images_cfg.max_per_dir)
        print(f"Target images: {len(target_imgs)}")

        print("Listing other images ...")
        all_imgs = ftp.walk_images(root, images_cfg.exts, images_cfg.recursive, images_cfg.max_per_dir)
        other_imgs = [p for p in all_imgs if not p.startswith(target_dir.rstrip("/") + "/") and p != target_dir]
        if limits_cfg.max_others_to_embed and len(other_imgs) > limits_cfg.max_others_to_embed:
            other_imgs = other_imgs[:limits_cfg.max_others_to_embed]
        print(f"Other images: {len(other_imgs)}")

        if len(target_imgs) == 0 or len(other_imgs) == 0:
            print("[ERROR] Need at least 1 target image and 1 other image.", file=sys.stderr)
            return

        # Download to cache
        print("Downloading target images to cache ...")
        target_locals = download_many(ftp, target_imgs, cache_cfg.dir)
        target_pairs = [(rp, lp) for rp, lp in zip(target_imgs, target_locals) if lp]
        if len(target_pairs) == 0:
            print("[ERROR] No target images downloaded successfully.", file=sys.stderr)
            return

        print("Downloading other images to cache ...")
        other_locals = download_many(ftp, other_imgs, cache_cfg.dir)
        other_pairs = [(rp, lp) for rp, lp in zip(other_imgs, other_locals) if lp]
        if len(other_pairs) == 0:
            print("[ERROR] No other images downloaded successfully.", file=sys.stderr)
            return

        # Load model
        device = emb_cfg.device if (emb_cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"
        print(f"Loading OpenCLIP model on {device} ...")
        model, preprocess = load_openclip("ViT-H-14", "laion2b_s32b_b79k", device)

        # Compute embeddings
        print("Embedding target images ...")
        target_embs = batched_embed([lp for _, lp in target_pairs], preprocess, model, device, emb_cfg.batch_size)
        # Drop failed rows (rare)
        valid_t = [i for i, e in enumerate(target_embs) if e is not None]
        target_embs = np.stack([target_embs[i] for i in valid_t], axis=0)

        print("Embedding other images ...")
        other_embs = batched_embed([lp for _, lp in other_pairs], preprocess, model, device, emb_cfg.batch_size)
        valid_o = [i for i, e in enumerate(other_embs) if e is not None]
        other_embs = np.stack([other_embs[i] for i in valid_o], axis=0)
        other_pairs = [other_pairs[i] for i in valid_o]

        # Similarity scoring
        print("Computing average top-K similarities ...")
        scores = avg_topk_sim(other_embs, target_embs, emb_cfg.topk)  # numpy [M]
        # Rank high to low
        rank_idx = np.argsort(-scores)
        other_pairs = [other_pairs[i] for i in rank_idx]
        scores = scores[rank_idx]

        # Sampling 5 bands by rank percentiles
        M = len(other_pairs)
        band_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # expressed as fraction of list, 1.0 = best end
        # But our list is already sorted high->low. So bands: [0,0.2*M) is 100~80%, etc.
        random.seed(review_cfg.seed)
        sampled_indices = []
        for b in range(5):
            start = int(math.floor(band_edges[b] * M))
            end   = int(math.floor(band_edges[b+1] * M))
            # Boundaries: ensure at least 1 if possible
            bucket = list(range(start, min(end, M)))
            if not bucket:
                continue
            k = max(review_cfg.sample_per_band, int(round(0.1 * len(bucket))))
            sampled_indices += random.sample(bucket, min(k, len(bucket)))

        sampled_indices = sorted(sampled_indices)
        # Create sampling set and remaining set
        sampled_mask = np.zeros(M, dtype=bool)
        sampled_mask[sampled_indices] = True
        sampled_pairs = [other_pairs[i] for i in sampled_indices]
        sampled_scores = [float(scores[i]) for i in sampled_indices]

        remaining_pairs = [other_pairs[i] for i in range(M) if not sampled_mask[i]]
        remaining_scores = [float(scores[i]) for i in range(M) if not sampled_mask[i]]

        # Stage 1: review sampled 50
        cv2.namedWindow(display_cfg.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(display_cfg.window_name, display_cfg.default_window_wh[1], display_cfg.default_window_wh[0])

        moved_sim = []
        ignored_sim = []
        stage_rows = []

        print("\nStage 1: sampled review. Press 'j' to move to target, 'k' to ignore, 'esc' to abort.")
        for (rp, lp), sim in zip(sampled_pairs, sampled_scores):
            img = cv2.imread(lp)
            if img is None:
                print(f"[WARN] Cannot read {lp}")
                continue
            H, W = get_window_size(display_cfg.window_name, display_cfg.default_window_wh)
            vis = scale_to_fit(img, H, W)
            head = f"{rp} | sim={sim:.4f} | SAMPLE"
            vis2 = put_text_top(vis, head, display_cfg.font_scale, display_cfg.font_thick)
            cv2.imshow(display_cfg.window_name, vis2)
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC
                print("Aborted by user.")
                ftp.close()
                return
            if key == ord('j'):
                # Move on FTP
                try:
                    new_remote = ftp.move_file(rp, target_dir)
                    decision = "move"
                    moved_sim.append(sim)
                except Exception as e:
                    print(f"[ERROR] Move failed {rp}: {e}", file=sys.stderr)
                    new_remote = ""
                    decision = "move_failed"
                stage_rows.append((rp, new_remote, sim, decision, "sample"))
            else:
                decision = "ignore"
                ignored_sim.append(sim)
                stage_rows.append((rp, "", sim, decision, "sample"))

        log_csv_append(log_cfg.csv_path, stage_rows)

        # Compute threshold from sample
        if moved_sim or ignored_sim:
            x = [*moved_sim, *ignored_sim]
            y = [1]*len(moved_sim) + [0]*len(ignored_sim)
            t = weighted_threshold(x, y)
            lo_moved = min(moved_sim) if moved_sim else float("nan")
            hi_ignored = max(ignored_sim) if ignored_sim else float("nan")
            print("\n--- Threshold summary (from sampled decisions) ---")
            print(f"Lowest similarity MOVED   : {lo_moved if moved_sim else 'n/a'}")
            print(f"Highest similarity IGNORED: {hi_ignored if ignored_sim else 'n/a'}")
            print(f"Weighted cut threshold    : {t:.6f}")
        else:
            t = float("nan")
            print("\nNo labeled decisions collected in sample; threshold unavailable.")

        # Stage 2: review remaining items high->low
        print("\nStage 2: remaining review high→low. Press 'j' move, 'k' ignore, 'esc' abort.")
        rows2 = []
        for (rp, lp), sim in zip(remaining_pairs, remaining_scores):
            img = cv2.imread(lp)
            if img is None:
                print(f"[WARN] Cannot read {lp}")
                continue
            H, W = get_window_size(display_cfg.window_name, display_cfg.default_window_wh)
            vis = scale_to_fit(img, H, W)
            hint = f"{rp} | sim={sim:.4f} | TH={t:.4f}" if math.isfinite(t) else f"{rp} | sim={sim:.4f}"
            vis2 = put_text_top(vis, hint, display_cfg.font_scale, display_cfg.font_thick)
            cv2.imshow(display_cfg.window_name, vis2)
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                print("Done.")
                break
            if key == ord('j'):
                try:
                    new_remote = ftp.move_file(rp, target_dir)
                    decision = "move"
                except Exception as e:
                    print(f"[ERROR] Move failed {rp}: {e}", file=sys.stderr)
                    new_remote = ""
                    decision = "move_failed"
            else:
                decision = "ignore"
                new_remote = ""
            rows2.append((rp, new_remote, sim, decision, "remaining"))
        if rows2:
            log_csv_append(log_cfg.csv_path, rows2)

        cv2.destroyAllWindows()
        ftp.close()
        print(f"\nLogs written to: {log_cfg.csv_path}")

    finally:
        try:
            ftp.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
