# -*- coding: utf-8 -*-
"""
Interactive similarity triage for Android images via ADB.

Features
- Read config.yml or a path passed by --config.
- CSV schema: [relative_path, cached_image_path_unused, emb_npy_path]
- Define target set by config.index_folder prefixes.
- Skip candidates in config.skip_folder prefixes.
- Compute average of top-10 cosine similarities vs targets.
- Sort candidates by score. Randomly sample 5 percentile bands
  (100–80, 80–60, 60–40, 40–20, 20–0). Up to 10 per band, total ≤ 50.
- Interactive review:
    - Only pull the image from phone when displaying.
    - Resize to fit current OpenCV window without aspect distortion.
    - j = move to target folder on device (conflict-safe rename).
    - k = keep in place.
- After sampled set:
    - Report lowest similarity MOVED, highest similarity IGNORED,
      and threshold = midpoint of those two.
    - Continue reviewing remaining items from high→low similarity.
- GPU acceleration via PyTorch if available.
- Progress bar during similarity computation.
- Prints current index during manual selection.

Assumptions
- Embeddings are saved as .npy vectors. We renormalize for safety.
- If config lacks explicit target_move_folder, default to index_folder[0].
- Relative paths in CSV start with "/" and are relative to /storage/emulated/0.

Dependencies:
    pip install adbutils pyyaml numpy opencv-python torch tqdm
"""

import os
import sys
import csv
import math
import time
import random
import shutil
import tempfile
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import yaml
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from adbutils import adb

# ---------------- Config and I/O ----------------

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # Defaults
    cfg.setdefault("local_root", ".")
    cfg.setdefault("adb_cache_dir", "_adb_cache")
    cfg.setdefault("emb_cache_dir", "_emb_cache")
    cfg.setdefault("csv_path", "index.csv")
    cfg.setdefault("index_folder", [])
    cfg.setdefault("skip_folder", [])
    # Optional explicit move target; else use first index_folder
    if "target_move_folder" not in cfg:
        if cfg["index_folder"]:
            cfg["target_move_folder"] = cfg["index_folder"][0]
        else:
            raise ValueError("Config requires index_folder or target_move_folder.")
    return cfg

def read_csv_rows(csv_path: Path) -> List[Tuple[str, str, str]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)  # skip header
        for line in r:
            if not line or len(line) < 3:
                continue
            rel_path = line[0].strip()
            cache_img = line[1].strip()
            emb_path = line[2].strip()
            rows.append((rel_path, cache_img, emb_path))
    return rows

def update_csv_path(csv_path: Path, rel_old: str, rel_new: str):
    """Replace the first-column path in the CSV when an image is moved."""
    tmp_path = csv_path.with_suffix(".tmp")
    with open(csv_path, "r", encoding="utf-8", newline="") as fin, \
         open(tmp_path, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        for i, row in enumerate(reader):
            if i == 0 or len(row) < 1:
                writer.writerow(row)
                continue
            if row[0].strip() == rel_old:
                row[0] = rel_new
            writer.writerow(row)
    os.replace(tmp_path, csv_path)

# ---------------- Path filters ----------------

def prefix_match(path: str, prefixes: List[str]) -> bool:
    # Normalize to single forward-slash style. CSV uses forward slashes already.
    p = path.rstrip("/")
    for pre in prefixes:
        pre = pre.rstrip("/")
        if p == pre or p.startswith(pre + "/"):
            return True
    return False

# ---------------- Embeddings ----------------

def load_embeddings(rows: List[Tuple[str, str, str]], emb_dir) -> Dict[str, np.ndarray]:
    """
    Load all embeddings in memory once. Keyed by relative path.
    """
    out = {}
    for rel, _, emb in rows:
        try:
            vec = np.load(os.path.join(emb_dir, emb), allow_pickle=False)
            vec = np.asarray(vec, dtype=np.float32)
            out[rel] = vec
        except Exception as e:
            print(f"[WARN] failed to load emb for {rel}: {emb} ({e})", file=sys.stderr)
    return out

def torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # mac
        return torch.device("mps")
    return torch.device("cpu")

def normalize_torch(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))

# ---------------- Similarity scoring ----------------

def compute_scores(
    all_rows: List[Tuple[str, str, str]],
    embs: Dict[str, np.ndarray],
    index_prefixes: List[str],
    skip_prefixes: List[str],
    topk: int = 10,
    batch: int = 2048,
) -> Tuple[List[Tuple[str, float]], List[str], torch.Tensor]:
    """
    Returns:
      sorted_scores: list of (rel_path, avg_topk_sim) for candidates
      target_list: list of target rel_paths used
      target_tensor: normalized torch tensor of target embeddings [N,D] on device
    """
    # Partition rows
    targets = []
    candidates = []
    for rel, _, _ in all_rows:
        if rel not in embs:
            continue
        if prefix_match(rel, index_prefixes):
            targets.append(rel)
        elif not prefix_match(rel, skip_prefixes):
            candidates.append(rel)

    if not targets:
        raise ValueError("No target images found under index_folder prefixes.")

    dev = torch_device()
    # Build target tensor
    tvecs = [embs[r] for r in targets]
    t = torch.from_numpy(np.stack(tvecs, axis=0))  # [N,D], float32
    t = normalize_torch(t).to(dev, non_blocking=True)
    tT = t.transpose(0, 1).contiguous()  # for matmul if needed

    # Score candidates
    scores = []
    with torch.no_grad():
        for i in tqdm(range(0, len(candidates), batch), desc="Scoring", unit="batch"):
            chunk_rel = candidates[i : i + batch]
            x = [embs[r] for r in chunk_rel]
            x = torch.from_numpy(np.stack(x, axis=0))
            x = normalize_torch(x).to(dev, non_blocking=True)  # [B,D]
            # Cosine sim = dot because both normalized
            # Compute sims: [B,N_targets]
            sims = x @ tT  # matmul
            # topk average for each row
            k = min(topk, sims.shape[1])
            vals, _ = torch.topk(sims, k=k, dim=1, largest=True, sorted=False)
            avg = vals.mean(dim=1).float().cpu().tolist()
            scores.extend(zip(chunk_rel, avg))
            del x, sims, vals
    # Sort high to low
    scores.sort(key=lambda z: z[1], reverse=True)
    return scores, targets, t

# ---------------- Sampling by percentile bands ----------------

def stratified_sample(scores: List[Tuple[str, float]], per_band: int = 15) -> List[Tuple[str, float]]:
    """
    Stratified sampling by rank percentiles (high → low).
    Scores must be sorted descending by similarity.
    Bands (percentile ranges of rank):
        [95–100%), [90–95%), [80–90%), [50–80%), [0–50%)
    Randomly sample up to `per_band` items per band.
    Return keeps the same descending order as input.
    """
    n = len(scores)
    if n == 0:
        return []

    # descending rank-based bands, expressed as fraction of total rank
    bands = [
        (0.95, 1.00),  # top 5%
        (0.90, 0.95),  # 90–95%
        (0.80, 0.90),  # 80–90%
        (0.50, 0.80),  # 50–80%
        (0.00, 0.50),  # bottom 50%
    ]

    out = []
    for lo, hi in bands:
        # translate rank percentiles into indices
        start = int(round((1.0 - hi) * n))      # inclusive
        end   = int(round((1.0 - lo) * n))      # exclusive
        segment = scores[start:end]
        if not segment:
            continue
        k = min(per_band, len(segment))
        out.extend(random.sample(segment, k))

    # preserve input descending order
    picked = {r for r, _ in out}
    return [(r, s) for (r, s) in scores if r in picked]

# ---------------- ADB helpers ----------------

STORAGE_ROOT = "/storage/emulated/0"

def pick_device():
    devs = adb.device_list()
    if not devs:
        raise RuntimeError("No ADB device connected.")
    # Choose the first device. Adjust if you need selection.
    return devs[0]

def device_path(rel_path: str) -> str:
    # rel_path example: "/UserFile/a.jpg"
    if not rel_path.startswith("/"):
        raise ValueError(f"CSV relative path must start with '/': {rel_path}")
    return STORAGE_ROOT + rel_path

def adb_pull_temp(dev, rel_path: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dpath = device_path(rel_path)
    tmp = tempfile.NamedTemporaryFile(prefix="._adb_", suffix=Path(rel_path).suffix or ".jpg", dir=dest_dir, delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        dev.sync.pull(dpath, str(tmp_path))
        return tmp_path
    except Exception as e:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise RuntimeError(f"ADB pull failed: {rel_path} ({e})")

def adb_move_conflict_safe(dev, src_rel: str, dst_dir_rel: str) -> str:
    """
    Move file on device from src_rel to dst_dir_rel with conflict-safe rename.
    Returns final destination relative path.
    """
    src_abs = device_path(src_rel)
    # Ensure destination directory exists
    dst_dir_abs = device_path(dst_dir_rel)
    dev.shell(f"mkdir -p {sh_quote(dst_dir_abs)}")

    base = Path(src_rel).name
    stem = Path(base).stem
    suf = Path(base).suffix
    # Try direct name then add -1, -2, ...
    attempt = 0
    while True:
        candidate_name = base if attempt == 0 else f"{stem}-{attempt}{suf}"
        dst_rel = join_rel(dst_dir_rel, candidate_name)
        dst_abs = device_path(dst_rel)
        # Test existence on device
        res = dev.shell2(f"test -e {sh_quote(dst_abs)}")
        if res.returncode != 0:
            # not exists, safe to move
            mv_res = dev.shell2(f"mv {sh_quote(src_abs)} {sh_quote(dst_abs)}")
            if mv_res.returncode != 0:
                raise RuntimeError(f"ADB mv failed: {mv_res.output or mv_res.stderr}")
            return dst_rel
        attempt += 1

def sh_quote(p: str) -> str:
    # Minimal single-quote safe
    return "'" + p.replace("'", "'\"'\"'") + "'"

def join_rel(dir_rel: str, name: str) -> str:
    dir_rel = dir_rel.rstrip("/")
    if not dir_rel.startswith("/"):
        raise ValueError(f"dest dir must be relative to {STORAGE_ROOT} and start with '/': {dir_rel}")
    return dir_rel + "/" + name

# ---------------- Display helpers ----------------

def get_window_size(win: str, default_wh=(900, 1200)) -> Tuple[int, int]:
    try:
        # getWindowImageRect returns (x,y,w,h) on many OpenCV builds
        x, y, w, h = cv2.getWindowImageRect(win)
        if w > 0 and h > 0:
            return (h, w)  # return as (H, W)
    except Exception:
        pass
    # Fallback to default
    return (default_wh[0], default_wh[1])

def fit_image(img: np.ndarray, win_h: int, win_w: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    if ih == 0 or iw == 0:
        return img
    scale = min(win_w / iw, win_h / ih)
    new_w = max(1, int(round(iw * scale)))
    new_h = max(1, int(round(ih * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    # Letterbox to window
    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    off_y = (win_h - new_h) // 2
    off_x = (win_w - new_w) // 2
    canvas[off_y:off_y+new_h, off_x:off_x+new_w] = resized
    return canvas

def draw_banner(frame: np.ndarray, text_top: str, text_bottom: str = "") -> None:
    # Simple readable overlay
    h, w = frame.shape[:2]
    pad = 6
    # Top
    top_bg_h = 26
    cv2.rectangle(frame, (0, 0), (w, top_bg_h), (0, 0, 0), thickness=-1)
    cv2.putText(frame, text_top, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    # Bottom
    if text_bottom:
        bot_bg_h = 26
        cv2.rectangle(frame, (0, h - bot_bg_h), (w, h), (0, 0, 0), thickness=-1)
        cv2.putText(frame, text_bottom, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

def show_and_decide(
    dev,
    rel_path: str,
    sim: float,
    win: str,
    cache_dir: Path,
    idx_info: str,
    bottom_hint: str,
) -> str:
    """
    Returns: 'move', 'skip', 'back', or 'quit'.
    """
    local_path = None
    try:
        local_path = adb_pull_temp(dev, rel_path, cache_dir)
        img = cv2.imread(str(local_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Unable to read image: {rel_path}", file=sys.stderr)
            return 'skip'
        while True:
            h, w = get_window_size(win)
            frame = fit_image(img, h, w)
            draw_banner(frame, f"{rel_path}  |  sim={sim:.4f}", bottom_hint)
            cv2.imshow(win, frame)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('j'):
                return 'move'
            if key == ord('k'):
                return 'skip'
            if key == 27:  # ESC
                return 'quit'
            if key == ord('f'):
                return 'back'
    finally:
        try:
            if local_path and Path(local_path).exists():
                Path(local_path).unlink()
        except Exception:
            pass

# ---------------- Main workflow ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yml", help="Path to YAML config")
    ap.add_argument("--topk", type=int, default=10, help="Top-K average for scoring")
    ap.add_argument("--per_band", type=int, default=15, help="Sample count per percentile band")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for sampling")
    args = ap.parse_args()

    random.seed(args.seed)

    cfg = load_config(Path(args.config))
    local_root = Path(cfg["local_root"]).resolve()
    adb_cache_dir = local_root / cfg["adb_cache_dir"]
    emb_cache_dir = local_root / cfg["emb_cache_dir"]
    csv_path = local_root / cfg["csv_path"]
    index_prefixes = cfg["index_folder"]
    skip_prefixes = cfg["skip_folder"]
    move_target_rel_dir = cfg["target_move_folder"]

    rows = read_csv_rows(csv_path)
    if not rows:
        print("No CSV rows.", file=sys.stderr)
        sys.exit(1)

    # Load embeddings
    embs = load_embeddings(rows, emb_cache_dir)

    # Compute scores with GPU if available
    scores, targets, tmat = compute_scores(
        rows, embs, index_prefixes, skip_prefixes, topk=args.topk
    )
    print(f"Candidates to review: {len(scores)}; Targets: {len(targets)}")

    # Stratified sample
    sampled = stratified_sample(scores, per_band=args.per_band)
    sampled_ids = set([r for r, _ in sampled])
    print(f"Sampled for manual triage: {len(sampled)}")

    # Interactive review
    dev = pick_device()
    win = "Review"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(win, 900, 1200)

    moved_scores = []
    ignored_scores = []

    # Helper to print progress index
    def print_idx(cur, total, stage: str):
        print(f"[{stage}] {cur}/{total}")

    # Stage 1: sampled set
    for i, (rel, sim) in enumerate(sampled, 1):
        print_idx(i, len(sampled), "sampled")
        bottom = "Press j=move, k=skip, ESC=quit"
        act = show_and_decide(dev, rel, sim, win, adb_cache_dir, f"{i}/{len(sampled)}", bottom)
        if act == 'quit':
            cv2.destroyAllWindows()
            sys.exit(0)
        if act == 'move':
            try:
                dst_rel = adb_move_conflict_safe(dev, rel, move_target_rel_dir)
                moved_scores.append((rel, sim, dst_rel))
                print(f"[MOVE] {rel} -> {dst_rel}  sim={sim:.4f}")
                try:
                    update_csv_path(csv_path, rel, dst_rel)
                except Exception as e:
                    print(f"[WARN] failed to update CSV for {rel}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"[ERROR] move failed for {rel}: {e}", file=sys.stderr)
        else:
            ignored_scores.append((rel, sim))
            print(f"[SKIP] {rel}  sim={sim:.4f}")

    # Threshold report
    lowest_moved = min((s for (_, s, __) in moved_scores), default=float('nan'))
    highest_ignored = max((s for (_, s) in ignored_scores), default=float('nan'))
    if not math.isnan(lowest_moved) and not math.isnan(highest_ignored):
        threshold = 0.5 * (lowest_moved + highest_ignored)
    else:
        threshold = float('nan')
    print("\n--- Sampled summary ---")
    print(f"Lowest similarity MOVED:   {lowest_moved if not math.isnan(lowest_moved) else 'N/A'}")
    print(f"Highest similarity IGNORED:{highest_ignored if not math.isnan(highest_ignored) else 'N/A'}")
    print(f"Suggested threshold:       {threshold if not math.isnan(threshold) else 'N/A'}")

    # Stage 2: remaining items from high to low
    remaining = [(r, s) for (r, s) in scores if r not in sampled_ids]
    print(f"\nRemaining to review: {len(remaining)}")
    i = 0
    history = []  # store skipped indices for back navigation

    while 0 <= i < len(remaining):
        rel, sim = remaining[i]
        print_idx(i + 1, len(remaining), "remaining")
        top_hint = f"sim={sim:.4f}  threshold={threshold if not math.isnan(threshold) else 'N/A'}"
        bottom = "Press j=move, k=skip, f=back, ESC=quit"
        act = show_and_decide(dev, rel, sim, win, adb_cache_dir, f"{i + 1}/{len(remaining)}", bottom)

        if act == 'quit':
            break
        elif act == 'back':
            if history:
                i = history.pop()  # go back to last skipped
                continue
            else:
                print("[INFO] No previous skipped image.")
                continue
        elif act == 'move':
            try:
                dst_rel = adb_move_conflict_safe(dev, rel, move_target_rel_dir)
                moved_scores.append((rel, sim, dst_rel))
                print(f"[MOVE] {rel} -> {dst_rel}  sim={sim:.4f}")
                update_csv_path(csv_path, rel, dst_rel)
            except Exception as e:
                print(f"[ERROR] move failed for {rel}: {e}", file=sys.stderr)
        else:  # skip
            ignored_scores.append((rel, sim))
            history.append(i)  # remember skipped index for potential backtrack
            print(f"[SKIP] {rel}  sim={sim:.4f}")

        i += 1

    cv2.destroyAllWindows()
    print("\nDone.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        sys.exit(130)
