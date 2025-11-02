#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADB image sampler + mover with GPU similarity and mtime cache.

Deps:
  pip install adbutils pyyaml numpy torch opencv-python tqdm
"""

import os
import sys
import io
import csv
import math
import time
import argparse
import random
import tempfile
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Iterable

import yaml
import numpy as np
from tqdm import tqdm

try:
    import torch
    TORCH_OK = True
except Exception as _e:
    print(f"[WARN] torch not usable: {_e}", file=sys.stderr)
    TORCH_OK = False

import cv2
from adbutils import adb

# ----------------------------- Config & Data -----------------------------

@dataclass
class Config:
    index_folder: List[str]
    skip_folder: List[str]
    local_root: str
    adb_cache_dir: str
    emb_cache_dir: str
    csv_path: str
    per_band: int = 15
    seed: int = 20251102

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        return Config(
            index_folder=list(y.get("index_folder", [])),
            skip_folder=list(y.get("skip_folder", [])),
            local_root=y.get("local_root", "."),
            adb_cache_dir=y.get("adb_cache_dir", "_adb_cache"),
            emb_cache_dir=y.get("emb_cache_dir", "_emb_cache"),
            csv_path=y.get("csv_path", "index.csv"),
            per_band=int(y.get("per_band", 15)),
            seed=int(y.get("seed", 20251102)),
        )

@dataclass
class RowRec:
    row_idx: int
    rel_path: str
    emb_rel_or_abs: str
    emb_abs: str
    in_index: bool
    in_skip: bool

@dataclass
class Item:
    rec: RowRec
    vec: np.ndarray
    idx_sim: float
    time_close_bonus: bool
    sim_with_bonus: float
    decided: str = "pending"
    moved_new_rel: Optional[str] = None

# ----------------------------- ADB Helpers ------------------------------

STORAGE_PREFIX = "/storage/emulated/0"
MTIME_CACHE_NAME = "mtimes.json"

def choose_device():
    devs = adb.device_list()
    if not devs:
        raise RuntimeError("No ADB device found. Check 'adb devices'.")
    return devs[0]

def abs_remote(rel_path: str) -> str:
    if not rel_path.startswith("/"):
        rel_path = "/" + rel_path
    return STORAGE_PREFIX + rel_path

def remote_exists(dev, abs_path: str) -> bool:
    out = dev.shell(f'[ -e "{abs_path}" ] && echo 1 || echo 0').strip()
    return out == "1"

def ensure_remote_dir(dev, abs_dir: str):
    dev.shell(f'mkdir -p "{abs_dir}"')

def get_remote_mtime(dev, abs_path: str) -> Optional[int]:
    cmds = [
        f'stat -c %Y "{abs_path}"',
        f'toybox stat -c %Y "{abs_path}"',
        f'busybox stat -c %Y "{abs_path}"',
    ]
    for cmd in cmds:
        out = dev.shell(cmd).strip()
        if out.isdigit():
            return int(out)
    return None

def pull_temp_image(dev, rel_path: str, local_tmp_dir: str) -> Optional[str]:
    abs_path = abs_remote(rel_path)
    if not remote_exists(dev, abs_path):
        print(f"[WARN] remote missing: {rel_path}", file=sys.stderr)
        return None
    fd, tmp_path = tempfile.mkstemp(prefix="adbimg_", suffix=os.path.splitext(rel_path)[1], dir=local_tmp_dir)
    os.close(fd)
    try:
        dev.sync.pull(abs_path, tmp_path)
        return tmp_path
    except Exception as e:
        print(f"[WARN] pull failed: {rel_path} ({e})", file=sys.stderr)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return None

def move_remote_with_rename(dev, src_rel: str, dest_dir_rel: str) -> Optional[str]:
    src_abs = abs_remote(src_rel)
    base = os.path.basename(src_rel)
    name, ext = os.path.splitext(base)
    dest_dir_abs = abs_remote(dest_dir_rel.rstrip("/"))
    ensure_remote_dir(dev, dest_dir_abs)
    candidate = base
    for i in range(0, 1000):
        dest_abs = dest_dir_abs + "/" + candidate
        if not remote_exists(dev, dest_abs):
            res = dev.shell(f'mv "{src_abs}" "{dest_abs}" 2>/dev/null; echo $?').strip()
            if res == "0":
                return dest_dir_rel.rstrip("/") + "/" + candidate
            cp = dev.shell(f'cp "{src_abs}" "{dest_abs}" 2>/dev/null; echo $?').strip()
            if cp == "0":
                rm = dev.shell(f'rm "{src_abs}" 2>/dev/null; echo $?').strip()
                if rm == "0":
                    return dest_dir_rel.rstrip("/") + "/" + candidate
        candidate = f"{name}-{i+1}{ext}"
    print(f"[ERROR] cannot move with rename after many attempts: {src_rel} -> {dest_dir_rel}", file=sys.stderr)
    return None

# ----------------------------- Mtime Cache -------------------------------

def mtime_cache_path(local_root: str) -> str:
    return os.path.join(local_root, MTIME_CACHE_NAME)

def mtime_cache_load(local_root: str) -> Dict[str, int]:
    p = mtime_cache_path(local_root)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        out: Dict[str, int] = {}
        for k, v in d.items():
            try:
                out[str(k)] = int(v)
            except Exception:
                pass
        return out
    except Exception:
        return {}

def mtime_cache_save(local_root: str, cache: Dict[str, int]) -> None:
    p = mtime_cache_path(local_root)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, p)

def get_remote_mtime_cached(dev, rel_path: str, cache: Dict[str, int]) -> Optional[int]:
    if rel_path in cache:
        return cache[rel_path]
    mt = get_remote_mtime(dev, abs_remote(rel_path))
    if mt is not None:
        cache[rel_path] = int(mt)
    return mt

def mtime_cache_on_move(cache: Dict[str, int], old_rel: str, new_rel: str, dev=None):
    v = cache.pop(old_rel, None)
    if dev is not None:
        new_mt = get_remote_mtime(dev, abs_remote(new_rel))
        if new_mt is not None:
            cache[new_rel] = int(new_mt)
            return
    if v is not None:
        cache[new_rel] = int(v)

# ----------------------------- IO Helpers -------------------------------

def resolve_emb_abs(emb_rel_or_abs: str, emb_cache_dir: str, local_root: str) -> str:
    p = emb_rel_or_abs
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(local_root, emb_cache_dir, p))

def load_csv_rows(csv_abs: str) -> Tuple[List[str], List[List[str]]]:
    with open(csv_abs, "r", encoding="utf-8", newline="") as f:
        reader = list(csv.reader(f))
    if not reader:
        raise RuntimeError("CSV empty")
    header = reader[0]
    rows = reader[1:]
    return header, rows

def save_csv_rows(csv_abs: str, header: List[str], rows: List[List[str]]):
    tmp = csv_abs + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, csv_abs)

def is_under_any(rel_path: str, prefixes: Iterable[str]) -> bool:
    for p in prefixes:
        if rel_path.startswith(p.rstrip("/")):
            return True
    return False

def load_vec(path: str) -> np.ndarray:
    v = np.load(path)
    v = v.astype(np.float32, copy=False).reshape(-1)
    n = np.linalg.norm(v)
    if n == 0 or not np.isfinite(n):
        return v
    return v / n

# ---------------------------- Similarities ------------------------------

def to_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x).to(device)

def compute_idx_sim_topk_mean(
    query_vecs: np.ndarray,
    index_vecs: np.ndarray,
    k: int,
    device: Optional[torch.device]
) -> np.ndarray:
    if index_vecs.shape[0] == 0:
        return np.zeros((query_vecs.shape[0],), dtype=np.float32)
    k = min(k, index_vecs.shape[0])
    M, _ = query_vecs.shape
    out = np.zeros((M,), dtype=np.float32)
    if device is None or not TORCH_OK:
        idxT = index_vecs.T
        for i in tqdm(range(M), desc="idx-sim (CPU)", unit="q"):
            s = query_vecs[i].dot(idxT)
            topk = np.partition(s, -k)[-k:]
            out[i] = float(topk.mean())
        return out
    with torch.no_grad():
        q = to_torch(query_vecs, device)
        idx = to_torch(index_vecs, device).t()
        B = 4096 if device.type == "cuda" else 1024
        for start in tqdm(range(0, M, B), desc="idx-sim (GPU)", unit="qb"):
            end = min(M, start + B)
            sims = q[start:end] @ idx
            topk_vals, _ = torch.topk(sims, k=k, dim=1, largest=True, sorted=False)
            out[start:end] = topk_vals.mean(dim=1).float().cpu().numpy()
    return out

def compute_any_within_mins(q_mt: np.ndarray, idx_mt: np.ndarray, mins: float) -> np.ndarray:
    thresh = int(mins * 60)
    if idx_mt.size == 0:
        return np.zeros_like(q_mt, dtype=bool)
    M = q_mt.shape[0]
    out = np.zeros((M,), dtype=bool)
    CH = 4096
    for s in tqdm(range(0, M, CH), desc="mtime check", unit="qb"):
        e = min(M, s + CH)
        q_chunk = q_mt[s:e][:, None]
        dif = np.abs(q_chunk - idx_mt[None, :])
        out[s:e] = (dif <= thresh).any(axis=1)
    return out

def compute_ignore_sim_topk_mean(
    query_vecs: np.ndarray,
    ignore_vecs: np.ndarray,
    k: int,
    device: Optional[torch.device]
) -> np.ndarray:
    M = query_vecs.shape[0]
    if ignore_vecs.shape[0] == 0:
        return np.ones((M,), dtype=np.float32)
    k = min(k, ignore_vecs.shape[0])
    if device is None or not TORCH_OK:
        ignT = ignore_vecs.T
        out = np.zeros((M,), dtype=np.float32)
        for i in range(M):
            s = query_vecs[i].dot(ignT)
            topk = np.partition(s, -k)[-k:]
            out[i] = float(topk.mean())
        return out
    with torch.no_grad():
        q = to_torch(query_vecs, device)
        ign = to_torch(ignore_vecs, device).t()
        out = np.zeros((M,), dtype=np.float32)
        B = 4096 if device.type == "cuda" else 1024
        for start in range(0, M, B):
            end = min(M, start + B)
            sims = q[start:end] @ ign
            topk_vals, _ = torch.topk(sims, k=k, dim=1, largest=True, sorted=False)
            out[start:end] = topk_vals.mean(dim=1).float().cpu().numpy()
        return out

# ----------------------------- UI Helpers --------------------------------

def fit_to_window(img: np.ndarray, win_w: int, win_h: int, fill_ratio: float = 0.90) -> np.ndarray:
    """
    Scale img so that either width or height reaches fill_ratio of the window.
    Keep aspect ratio. No stretching beyond that ratio.
    Example: win 700x1333, img 100x600 -> scale = 0.9 * min(700/100, 1333/600) = 2.0 -> 200x1200.
    """
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img
    # Best uniform scale to fit inside window
    scale_fit = min(win_w / float(w), win_h / float(h))
    scale = max(scale_fit * fill_ratio, 1e-6)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)

def put_banner(img: np.ndarray, lines: List[str]) -> np.ndarray:
    out = img.copy()
    y = 18
    for t in lines:
        cv2.putText(out, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        y += 22
    return out

# ----------------------------- Main Flow ----------------------------------

def stratified_sample_by_rank(sorted_items: List[Item], per_band: int, rng: random.Random) -> List[int]:
    n = len(sorted_items)
    bands = [
        (0.95, 1.00),
        (0.90, 0.95),
        (0.80, 0.90),
        (0.50, 0.80),
        (0.00, 0.50),
    ]
    out = []
    for lo, hi in bands:
        start = int(math.floor((1.0 - hi) * n))
        end   = int(math.floor((1.0 - lo) * n))
        if start >= end:
            continue
        idxs = list(range(start, end))
        rng.shuffle(idxs)
        out.extend(idxs[:per_band])
    out = sorted(set(out))
    return out

def main():
    ap = argparse.ArgumentParser(description="ADB image sampler and mover")
    ap.add_argument("--config", type=str, default="config.yml")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    rng = random.Random(cfg.seed)

    local_root = os.path.abspath(cfg.local_root)
    emb_root = os.path.normpath(os.path.join(local_root, cfg.emb_cache_dir))
    csv_abs = os.path.normpath(os.path.join(local_root, cfg.csv_path))
    tmp_view_dir = os.path.normpath(os.path.join(local_root, "_tmp_view"))
    os.makedirs(tmp_view_dir, exist_ok=True)

    # mtime cache
    mt_cache: Dict[str, int] = mtime_cache_load(local_root)

    header, rows = load_csv_rows(csv_abs)
    if len(header) < 3:
        raise RuntimeError("CSV header must have at least 3 columns")

    recs: List[RowRec] = []
    for i, row in enumerate(rows, start=1):
        rel = row[0].strip()
        emb_col = row[2].strip()
        emb_abs = resolve_emb_abs(emb_col, cfg.emb_cache_dir, local_root)
        in_index = is_under_any(rel, cfg.index_folder)
        in_skip  = is_under_any(rel, cfg.skip_folder)
        recs.append(RowRec(i, rel, emb_col, emb_abs, in_index, in_skip))

    index_recs = [r for r in recs if r.in_index]
    target_vecs, target_rel_paths = [], []
    for r in index_recs:
        if not os.path.exists(r.emb_abs):
            continue
        try:
            v = load_vec(r.emb_abs)
            target_vecs.append(v)
            target_rel_paths.append(r.rel_path)
        except Exception as e:
            print(f"[WARN] load vec failed: {r.emb_abs} ({e})", file=sys.stderr)
    if len(target_vecs) == 0:
        print("[ERROR] No target vectors found under index_folder. Abort.", file=sys.stderr)
        sys.exit(1)
    target_mat = np.stack(target_vecs, axis=0).astype(np.float32, copy=False)

    dev = choose_device()
    print(f"[INFO] Using device: {dev}", flush=True)

    print("[INFO] Collecting index mtimes ...")
    idx_mt = []
    for rel in tqdm(target_rel_paths, desc="stat index mtimes"):
        mt = get_remote_mtime_cached(dev, rel, mt_cache)
        idx_mt.append(-10**12 if mt is None else mt)
    idx_mt = np.array(idx_mt, dtype=np.int64)
    mtime_cache_save(local_root, mt_cache)

    cand_recs = [r for r in recs if not r.in_skip]
    cand_vecs, cand_map_idx = [], []
    for ci, r in enumerate(tqdm(cand_recs, desc="load candidate vecs")):
        if not os.path.exists(r.emb_abs):
            print(f"[WARN] missing vec: {r.emb_abs}", file=sys.stderr)
            continue
        try:
            cand_vecs.append(load_vec(r.emb_abs))
            cand_map_idx.append(ci)
        except Exception as e:
            print(f"[WARN] load vec failed: {r.emb_abs} ({e})", file=sys.stderr)
    if len(cand_vecs) == 0:
        print("[ERROR] No candidate vectors loaded. Abort.", file=sys.stderr)
        sys.exit(1)
    Q = np.stack(cand_vecs, axis=0).astype(np.float32, copy=False)

    device = None
    if TORCH_OK and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using CUDA: {torch.cuda.get_device_name(0)}")
    elif TORCH_OK:
        device = torch.device("cpu")
        print("[INFO] Using CPU torch")
    else:
        print("[INFO] Torch unavailable, CPU numpy fallback")

    idx_sims = compute_idx_sim_topk_mean(Q, target_mat, k=10, device=device)

    print("[INFO] Query mtimes ...")
    q_mt = []
    for ci in tqdm(cand_map_idx, desc="stat query mtimes"):
        rel = cand_recs[ci].rel_path
        mt = get_remote_mtime_cached(dev, rel, mt_cache)
        q_mt.append(-10**12 if mt is None else mt)
        if len(mt_cache) % 100 == 0:
            mtime_cache_save(local_root, mt_cache)
    q_mt = np.array(q_mt, dtype=np.int64)
    mtime_cache_save(local_root, mt_cache)

    flags = compute_any_within_mins(q_mt, idx_mt, mins=10)
    sim_with_bonus = np.where(flags, idx_sims * 1.15, idx_sims)

    items: List[Item] = []
    for pos, ci in enumerate(cand_map_idx):
        r = cand_recs[ci]
        items.append(Item(
            rec=r,
            vec=Q[pos],
            idx_sim=float(idx_sims[pos]),
            time_close_bonus=bool(flags[pos]),
            sim_with_bonus=float(sim_with_bonus[pos])
        ))

    items_sorted_idx = list(range(len(items)))
    items_sorted_idx.sort(key=lambda i: items[i].sim_with_bonus, reverse=True)

    sample_positions_in_sorted = stratified_sample_by_rank([items[i] for i in items_sorted_idx], cfg.per_band, rng)
    sample_global_idx = [items_sorted_idx[p] for p in sample_positions_in_sorted]

    win = "Review"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 700, 1200)

    ignored_set_indices: List[int] = []
    ignored_vecs_mat = np.zeros((0, items[0].vec.shape[0]), dtype=np.float32)
    skip_history: List[int] = []

    def overlay_lines(it: Item, ratio_value: Optional[float], phase: str):
        meta = [
            f"Phase: {phase}",
            f"IdxSim: {it.idx_sim:.4f}",
            f"Bonus30m: {'yes' if it.time_close_bonus else 'no'}",
            f"Score: {it.sim_with_bonus:.4f}" if ratio_value is None else f"Priority: {ratio_value:.4f}",
            f"Path: {it.rec.rel_path}",
            "j=move  k=ignore  f=back"
        ]
        return meta

    def get_window_size(win: str, default_wh=(900, 1200)) -> Tuple[int, int]:
        try:
            x, y, w, h = cv2.getWindowImageRect(win)
            if w > 0 and h > 0:
                return (h, w)
        except Exception:
            pass
        return (default_wh[0], default_wh[1])

    def fit_image(img: np.ndarray, win_h: int, win_w: int, fill_ratio: float = 0.9) -> np.ndarray:
        ih, iw = img.shape[:2]
        if ih == 0 or iw == 0:
            return img
        # fit 90% of the window (max dimension fills 90%)
        scale = fill_ratio * min(win_w / iw, win_h / ih)
        new_w = max(1, int(round(iw * scale)))
        new_h = max(1, int(round(ih * scale)))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        # center on black canvas
        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        off_y = (win_h - new_h) // 2
        off_x = (win_w - new_w) // 2
        canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
        return canvas

    def show_one(global_idx: int, pos: int, total: int, ratio_value: Optional[float], phase: str) -> str:
        it = items[global_idx]
        print(f"[{pos + 1}/{total}] {it.rec.rel_path}  sim={it.sim_with_bonus:.4f}", flush=True)
        tmp = pull_temp_image(dev, it.rec.rel_path, tmp_view_dir)
        if tmp is None:
            return "skip"
        img = cv2.imdecode(np.fromfile(tmp, dtype=np.uint8), cv2.IMREAD_COLOR)
        try:
            os.remove(tmp)
        except Exception:
            pass
        if img is None:
            print(f"[WARN] cannot decode image: {it.rec.rel_path}", file=sys.stderr)
            return "skip"

        while True:
            win_h, win_w = get_window_size(win)
            frame = fit_image(img, win_h, win_w)
            frame = put_banner(frame, overlay_lines(it, ratio_value, phase))
            cv2.imshow(win, frame)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('j'), ord('J')):
                return "move"
            if key in (ord('k'), ord('K')):
                return "ignore"
            if key in (ord('f'), ord('F')):
                return "back"
            if key == 27:  # ESC
                return "skip"
            # loop refreshes if user resizes window

    def try_add_to_ignored(global_idx: int):
        nonlocal ignored_vecs_mat
        if ignored_vecs_mat.shape[0] == 0:
            ignored_set_indices.append(global_idx)
            ignored_vecs_mat = np.concatenate([ignored_vecs_mat, items[global_idx].vec[None, :]], axis=0)
            return True
        v = items[global_idx].vec[None, :]
        sims = (ignored_vecs_mat @ v.T).reshape(-1)
        if np.all(sims <= 0.88 + 1e-6):
            ignored_set_indices.append(global_idx)
            ignored_vecs_mat = np.concatenate([ignored_vecs_mat, v], axis=0)
            return True
        return False

    def undo_last_skip():
        nonlocal ignored_vecs_mat
        if not skip_history:
            return None
        idx = skip_history.pop()
        it = items[idx]
        if it.decided == "ignored":
            if idx in ignored_set_indices:
                j = ignored_set_indices.index(idx)
                ignored_set_indices.pop(j)
                if ignored_vecs_mat.shape[0] > 0:
                    ignored_vecs_mat = np.delete(ignored_vecs_mat, j, axis=0)
            it.decided = "pending"
        return idx

    # ---------- Phase 1: sample ----------
    sample_moved_sims = []
    sample_ignored_sims = []

    for pi, gidx in enumerate(sample_global_idx):
        it = items[gidx]
        if it.decided != "pending":
            continue
        while True:
            action = show_one(gidx, pi, len(sample_global_idx), None, phase="sample")
            if action == "move":
                target_rel_dir = cfg.index_folder[0] if cfg.index_folder else os.path.dirname(it.rec.rel_path)
                new_rel = move_remote_with_rename(dev, it.rec.rel_path, target_rel_dir)
                if new_rel:
                    it.decided = "moved"
                    it.moved_new_rel = new_rel
                    rows[it.rec.row_idx - 1][0] = new_rel
                    save_csv_rows(csv_abs, header, rows)
                    # mtime cache update
                    mtime_cache_on_move(mt_cache, it.rec.rel_path, new_rel, dev=dev)
                    mtime_cache_save(local_root, mt_cache)
                    it.rec.rel_path = new_rel
                    sample_moved_sims.append(it.sim_with_bonus)
                else:
                    print(f"[WARN] move failed, keep pending: {it.rec.rel_path}", file=sys.stderr)
                break
            elif action == "ignore":
                it.decided = "ignored"
                skip_history.append(gidx)
                try_add_to_ignored(gidx)
                sample_ignored_sims.append(it.sim_with_bonus)
                break
            elif action == "back":
                prev = undo_last_skip()
                if prev is None:
                    print("[INFO] no skipped history")
                    continue
                gidx = prev
                it = items[gidx]
                continue
            else:
                it.decided = "ignored"
                skip_history.append(gidx)
                sample_ignored_sims.append(it.sim_with_bonus)
                break

    lowest_moved = min(sample_moved_sims) if sample_moved_sims else float("nan")
    biggest_ignored = max(sample_ignored_sims) if sample_ignored_sims else float("nan")
    threshold = 0.5 * (lowest_moved + biggest_ignored) if not (math.isnan(lowest_moved) or math.isnan(biggest_ignored)) else float("nan")
    print(f"[SAMPLE] lowest moved: {lowest_moved}")
    print(f"[SAMPLE] biggest ignored: {biggest_ignored}")
    print(f"[SAMPLE] threshold: {threshold}")

    # ---------- Phase 2: remaining ----------
    items_sorted_idx = list(range(len(items)))
    items_sorted_idx.sort(key=lambda i: items[i].sim_with_bonus, reverse=True)
    remaining_idx = [i for i in items_sorted_idx if items[i].decided == "pending"]
    if len(remaining_idx) == 0:
        print("[INFO] No remaining items. Done.")
        cv2.destroyWindow(win)
        return

    rem_vecs = np.stack([items[i].vec for i in remaining_idx], axis=0).astype(np.float32)
    rem_base_sim = np.array([items[i].sim_with_bonus for i in remaining_idx], dtype=np.float32)

    def recompute_priority_order() -> List[int]:
        if len(remaining_idx) == 0:
            return []
        if ignored_vecs_mat.shape[0] <= 1:
            order = list(range(len(remaining_idx)))
            order.sort(key=lambda j: rem_base_sim[j], reverse=True)
            return order
        dev_sel = torch.device("cuda") if (TORCH_OK and torch.cuda.is_available()) else (torch.device("cpu") if TORCH_OK else None)
        ign_mean = compute_ignore_sim_topk_mean(rem_vecs, ignored_vecs_mat, k=min(5, ignored_vecs_mat.shape[0]), device=dev_sel)
        ign_mean = np.maximum(ign_mean, 1e-4)
        pri = rem_base_sim / ign_mean
        order = list(range(len(remaining_idx)))
        order.sort(key=lambda j: pri[j], reverse=True)
        return order

    order = recompute_priority_order()

    i_ptr = 0
    while i_ptr < len(order):
        j = order[i_ptr]
        gidx = remaining_idx[j]
        it = items[gidx]

        ratio_val = None
        if ignored_vecs_mat.shape[0] > 1:
            dev_sel = torch.device("cuda") if (TORCH_OK and torch.cuda.is_available()) else (torch.device("cpu") if TORCH_OK else None)
            ign_mean_here = compute_ignore_sim_topk_mean(it.vec[None, :], ignored_vecs_mat, k=min(5, ignored_vecs_mat.shape[0]), device=dev_sel)[0]
            ign_mean_here = max(ign_mean_here, 1e-4)
            ratio_val = it.sim_with_bonus / ign_mean_here

        action = show_one(gidx, i_ptr, len(order), ratio_val, phase="main")
        if action == "move":
            target_rel_dir = cfg.index_folder[0] if cfg.index_folder else os.path.dirname(it.rec.rel_path)
            new_rel = move_remote_with_rename(dev, it.rec.rel_path, target_rel_dir)
            if new_rel:
                it.decided = "moved"
                it.moved_new_rel = new_rel
                rows[it.rec.row_idx - 1][0] = new_rel
                save_csv_rows(csv_abs, header, rows)
                mtime_cache_on_move(mt_cache, it.rec.rel_path, new_rel, dev=dev)
                mtime_cache_save(local_root, mt_cache)
                it.rec.rel_path = new_rel
                remaining_idx.pop(j)
                rem_vecs = np.delete(rem_vecs, j, axis=0)
                rem_base_sim = np.delete(rem_base_sim, j, axis=0)
                order = recompute_priority_order()
                i_ptr = 0
                continue
            else:
                print(f"[WARN] move failed, keep pending: {it.rec.rel_path}", file=sys.stderr)
                continue
        elif action == "ignore":
            it.decided = "ignored"
            skip_history.append(gidx)
            try_add_to_ignored(gidx)
            remaining_idx.pop(j)
            rem_vecs = np.delete(rem_vecs, j, axis=0)
            rem_base_sim = np.delete(rem_base_sim, j, axis=0)
            order = recompute_priority_order()
            i_ptr = 0
            continue
        elif action == "back":
            prev = undo_last_skip()
            if prev is None:
                print("[INFO] no skipped history")
                continue
            if prev not in remaining_idx and items[prev].decided == "pending":
                remaining_idx.append(prev)
                rem_vecs = np.concatenate([rem_vecs, items[prev].vec[None, :]], axis=0)
                rem_base_sim = np.concatenate([rem_base_sim, np.array([items[prev].sim_with_bonus], dtype=np.float32)], axis=0)
            order = recompute_priority_order()
            i_ptr = 0
            continue
        else:
            it.decided = "ignored"
            skip_history.append(gidx)
            remaining_idx.pop(j)
            rem_vecs = np.delete(rem_vecs, j, axis=0)
            rem_base_sim = np.delete(rem_base_sim, j, axis=0)
            order = recompute_priority_order()
            i_ptr = 0
            continue

    cv2.destroyWindow(win)
    print("[DONE] All selected images processed.")

if __name__ == "__main__":
    main()
