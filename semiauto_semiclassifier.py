import os
import sys
import cv2
import yaml
import math
import time
import ftplib
import random
import hashlib
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image

# ----------------- OpenCLIP -----------------
try:
    import open_clip
except Exception as e:
    print("Install OpenCLIP first: pip install open_clip_torch", file=sys.stderr)
    raise

# ----------------- Config types -----------------
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

@dataclass
class DisplayConfig:
    window_name: str = "Similarity Review"
    default_window_wh: Tuple[int, int] = (900, 1600)  # (H,W)
    font_scale: float = 0.6
    font_thick: int = 1

@dataclass
class LogConfig:
    csv_path: str = "./move_decisions.csv"

@dataclass
class LimitsConfig:
    max_others_to_embed: int = 0

# ----------------- Util -----------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def now_ts() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def is_image_name(name: str, exts: List[str]) -> bool:
    return os.path.splitext(name)[1].lower() in exts

def hash_path(remote_path: str) -> str:
    return hashlib.sha1(remote_path.encode("utf-8", errors="ignore")).hexdigest()[:12]

def build_local_cache_path(cache_dir: str, remote_path: str) -> str:
    base = os.path.basename(remote_path)
    h = hash_path(remote_path)
    name, ext = os.path.splitext(base)
    subdir = os.path.dirname(remote_path).strip("/").replace("/", "_")[:64]
    return os.path.join(cache_dir, subdir, f"{name}.{h}{ext}")

def log_csv_append(csv_path: str, rows: List[Tuple[str, str, float, str, str]]):
    header_needed = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("original_path,new_path,similarity,decision,stage,timestamp\n")
        for (orig, newp, sim, decision, stage) in rows:
            f.write(f"{orig},{newp},{sim:.6f},{decision},{stage},{now_ts()}\n")

# ----------------- Config load -----------------
def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

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

    tgt_cfg = cfg.get("target", {})
    target_folder = str(tgt_cfg.get("folder", "")).rstrip("/")
    ignore_dirs = [str(d).rstrip("/") for d in tgt_cfg.get("ignore_dirs", [])]

    return (ftp_cfg, cache_cfg, images_cfg, emb_cfg, review_cfg,
            display_cfg, log_cfg, limits_cfg, target_folder, ignore_dirs)

# ----------------- FTP client -----------------
class FTPClient:
    def __init__(self, cfg: FTPConfig):
        self.cfg = cfg
        self.ftp = ftplib.FTP()
        self.ftp.connect(cfg.host, cfg.port, timeout=30)
        # anonymous by default
        self.ftp.login()
        self.ftp.set_pasv(cfg.passive_mode)
        self.cwd(cfg.root_dir)

    def cwd(self, path: str):
        self.ftp.cwd(path)

    def pwd(self) -> str:
        return self.ftp.pwd()

    def close(self):
        try:
            self.ftp.quit()
        except Exception:
            try:
                self.ftp.close()
            except Exception:
                pass

    def list_dir(self, path: str) -> List[Tuple[str, bool]]:
        out: List[Tuple[str, bool]] = []
        try:
            self.ftp.cwd(path)
            for name, facts in self.ftp.mlsd():
                if name in (".", ".."):
                    continue
                is_dir = facts.get("type", "") == "dir"
                out.append((name, is_dir))
        except Exception:
            # fallback
            self.ftp.cwd(path)
            names = self.ftp.nlst()
            for full in names:
                name = os.path.basename(full)
                if name in (".", ".."):
                    continue
                is_dir = False
                try:
                    self.ftp.cwd(os.path.join(path, name))
                    is_dir = True
                    self.ftp.cwd(path)
                except Exception:
                    is_dir = False
                out.append((name, is_dir))
        return out

    def dir_exists(self, remote_dir: str) -> bool:
        parent = os.path.dirname(remote_dir.rstrip("/")) or "/"
        base = os.path.basename(remote_dir.rstrip("/"))
        try:
            for name, is_dir in self.list_dir(parent):
                if is_dir and name == base:
                    return True
        except Exception:
            return False
        return False

    def file_exists(self, remote_path: str) -> bool:
        parent = os.path.dirname(remote_path) or "/"
        base = os.path.basename(remote_path)
        try:
            for name, is_dir in self.list_dir(parent):
                if not is_dir and name == base:
                    return True
        except Exception:
            return False
        return False

    def download_file(self, remote_path: str, local_path: str):
        ensure_dir(os.path.dirname(local_path))
        with open(local_path, "wb") as f:
            self.ftp.retrbinary(f"RETR {remote_path}", f.write)

    def move_file(self, src: str, dst_dir: str) -> str:
        if not self.dir_exists(dst_dir):
            raise RuntimeError(f"FTP target dir not found: {dst_dir}")
        base = os.path.basename(src)
        stem, ext = os.path.splitext(base)
        candidate = os.path.join(dst_dir, base).replace("\\", "/")
        if not self.file_exists(candidate):
            self.ftp.rename(src, candidate)
            return candidate
        for i in range(1, 10000):
            alt = os.path.join(dst_dir, f"{stem}-{i}{ext}").replace("\\", "/")
            if not self.file_exists(alt):
                self.ftp.rename(src, alt)
                return alt
        raise RuntimeError("Auto-rename exceeded 9999 attempts")

    # basic walker without ignore logic (used for target)
    def walk_images(self, root: str, exts: List[str], recursive: bool, max_per_dir: int = 0) -> List[str]:
        stack = [root]
        files: List[str] = []
        while stack:
            d = stack.pop()
            try:
                entries = self.list_dir(d)
            except Exception as e:
                print(f"[WARN] list_dir failed at {d}: {e}", file=sys.stderr)
                continue
            cnt = 0
            for name, is_dir in entries:
                rp = os.path.join(d, name).replace("\\", "/")
                if is_dir:
                    if recursive:
                        stack.append(rp)
                else:
                    if is_image_name(name, exts):
                        files.append(rp)
                        cnt += 1
                        if max_per_dir and cnt >= max_per_dir:
                            break
        return sorted(files)

# walker with ignore pruning (used for "other" images)
def walk_images_filtered(ftp: FTPClient, root: str, exts: List[str], recursive: bool,
                         ignore_dirs: List[str], max_per_dir: int = 0) -> List[str]:
    stack = [root]
    files: List[str] = []
    ig_pref = [d.rstrip("/") + "/" for d in ignore_dirs]
    root_pref = root.rstrip("/") + "/"
    while stack:
        d = stack.pop()
        # skip ignored prefixes except the root itself
        if any(d.startswith(p) for p in ig_pref) and not d.startswith(root_pref):
            continue
        try:
            entries = ftp.list_dir(d)
        except Exception as e:
            print(f"[WARN] list_dir failed at {d}: {e}", file=sys.stderr)
            continue
        cnt = 0
        for name, is_dir in entries:
            rp = os.path.join(d, name).replace("\\", "/")
            if is_dir:
                if recursive:
                    if any(rp.startswith(p) for p in ig_pref):
                        # still allow walking if rp is the configured root itself
                        if not rp.startswith(root_pref):
                            continue
                    stack.append(rp)
            else:
                if is_image_name(name, exts):
                    files.append(rp)
                    cnt += 1
                    if max_per_dir and cnt >= max_per_dir:
                        break
    return sorted(files)

# ----------------- OpenCLIP helpers -----------------
def load_openclip(model_name: str, pretrained: str, device: str):
    dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained, device=dev)
    model.eval()
    return model, preprocess, dev

def pil_to_tensor(path: str, preprocess):
    img = Image.open(path).convert("RGB")
    return preprocess(img)

def batched_embed(paths: List[str], preprocess, model, device: str, batch_size: int):
    batch_tensors = []
    batch_indices = []
    out_chunks = []
    out_indices: List[int] = []
    for idx, p in enumerate(paths):
        try:
            t = pil_to_tensor(p, preprocess)
            batch_tensors.append(t)
            batch_indices.append(idx)
        except Exception as e:
            print(f"[WARN] failed to load {p}: {e}", file=sys.stderr)
        if len(batch_tensors) == batch_size or idx == len(paths) - 1:
            if batch_tensors:
                with torch.no_grad():
                    x = torch.stack(batch_tensors, dim=0).to(device)
                    feats = model.encode_image(x)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    feats = feats.detach().cpu().float().numpy()
                out_chunks.append(feats)
                out_indices.extend(batch_indices)
                batch_tensors, batch_indices = [], []
    if not out_chunks:
        return np.zeros((0, 1), dtype=np.float32), []
    embs = np.concatenate(out_chunks, axis=0)
    return embs, out_indices

def avg_topk_sim(other_embs: np.ndarray, target_embs: np.ndarray, topk: int) -> np.ndarray:
    if other_embs.size == 0 or target_embs.size == 0:
        return np.zeros((other_embs.shape[0] if other_embs.ndim == 2 else 0,), dtype=np.float32)
    with torch.no_grad():
        o = torch.from_numpy(other_embs)   # [M,D]
        t = torch.from_numpy(target_embs)  # [T,D]
        sims = o @ t.T                     # [M,T]
        k = min(topk, t.shape[0])
        vals = torch.topk(sims, k=k, dim=1).values
        mean_vals = vals.mean(dim=1)
    return mean_vals.cpu().numpy()

# ----------------- Threshold estimation -----------------
def weighted_threshold(similarities: List[float], decisions: List[int]) -> float:
    x = np.asarray(similarities, dtype=np.float64)
    y = np.asarray(decisions, dtype=np.int32)
    if len(x) == 0:
        return float("nan")
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float((x.min() + x.max()) / 2.0)

    w_pos = 0.5 / n_pos
    w_neg = 0.5 / n_neg

    order = np.argsort(x)
    xs = x[order]
    ys = y[order]

    wpos = (ys == 1).astype(float) * w_pos
    wneg = (ys == 0).astype(float) * w_neg
    cpos = np.cumsum(wpos)
    cneg = np.cumsum(wneg)
    total_neg = cneg[-1]

    best_err = float("inf")
    best_t = float((xs.min() + xs.max()) / 2.0)

    cuts = [-float("inf")] + [(xs[i] + xs[i+1]) / 2.0 for i in range(len(xs)-1)] + [float("inf")]
    for t in cuts:
        r = np.searchsorted(xs, t, side="left") - 1
        pos_left = cpos[r] if r >= 0 else 0.0
        neg_left = cneg[r] if r >= 0 else 0.0
        err = pos_left + (total_neg - neg_left)
        if err < best_err:
            best_err = err
            best_t = t
    if not np.isfinite(best_t):
        best_t = float((xs.min() + xs.max()) / 2.0)
    return float(best_t)

# ----------------- Display helpers -----------------
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
    if h <= 0 or w <= 0:
        return img
    s = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * s)))
    new_h = max(1, int(round(h * s)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def draw_top_text(img: np.ndarray, text: str, font_scale: float, thick: int):
    out = img.copy()
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out

# ----------------- Main -----------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python ftp_similarity_reviewer.py config.yml")
        sys.exit(1)

    (ftp_cfg, cache_cfg, images_cfg, emb_cfg, review_cfg,
     display_cfg, log_cfg, limits_cfg, target_dir, ignore_dirs) = load_config(sys.argv[1])

    if not target_dir:
        print("[ERROR] target.folder is required in config.yml", file=sys.stderr)
        sys.exit(2)

    ensure_dir(cache_cfg.dir)

    # FTP connect
    print(f"Connecting FTP {ftp_cfg.host}:{ftp_cfg.port}")
    try:
        ftp = FTPClient(ftp_cfg)
    except Exception as e:
        print(f"[ERROR] FTP connect/login failed: {e}", file=sys.stderr)
        sys.exit(3)

    try:
        if not ftp.dir_exists(target_dir):
            print(f"[ERROR] target folder not found on FTP: {target_dir}", file=sys.stderr)
            sys.exit(4)

        root = ftp_cfg.root_dir
        print(f"Target: {target_dir}")
        if ignore_dirs:
            print(f"Ignored prefixes: {ignore_dirs}")

        # List images
        print("Listing target images ...")
        target_imgs = ftp.walk_images(target_dir, images_cfg.exts, images_cfg.recursive, images_cfg.max_per_dir)
        print(f"Target images: {len(target_imgs)}")

        print("Listing other images ...")
        all_imgs = walk_images_filtered(ftp, root, images_cfg.exts, images_cfg.recursive, ignore_dirs, images_cfg.max_per_dir)
        other_imgs = [p for p in all_imgs if not p.startswith(target_dir.rstrip('/') + '/') and p != target_dir]
        if limits_cfg.max_others_to_embed and len(other_imgs) > limits_cfg.max_others_to_embed:
            other_imgs = other_imgs[:limits_cfg.max_others_to_embed]
        print(f"Other images: {len(other_imgs)}")

        if len(target_imgs) == 0 or len(other_imgs) == 0:
            print("[ERROR] Need >=1 target image and >=1 other image.", file=sys.stderr)
            sys.exit(5)

        # Download only what we need
        def download_many(remote_paths: List[str]) -> List[str]:
            locals_ = []
            for rp in remote_paths:
                lp = build_local_cache_path(cache_cfg.dir, rp)
                if not os.path.exists(lp):
                    try:
                        ftp.download_file(rp, lp)
                    except Exception as e:
                        print(f"[WARN] download failed {rp}: {e}", file=sys.stderr)
                        lp = ""
                locals_.append(lp)
            return locals_

        print("Downloading target images ...")
        target_locals = download_many(target_imgs)
        tgt_pairs = [(rp, lp) for rp, lp in zip(target_imgs, target_locals) if lp]
        if not tgt_pairs:
            print("[ERROR] no target image downloaded successfully.", file=sys.stderr)
            sys.exit(6)

        print("Downloading other images ...")
        other_locals = download_many(other_imgs)
        oth_pairs = [(rp, lp) for rp, lp in zip(other_imgs, other_locals) if lp]
        if not oth_pairs:
            print("[ERROR] no other image downloaded successfully.", file=sys.stderr)
            sys.exit(7)

        # Model
        print("Loading OpenCLIP ViT-H-14 laion2b_s32b_b79k ...")
        model, preprocess, device = load_openclip("ViT-H-14", "laion2b_s32b_b79k", emb_cfg.device)

        # Embeddings
        print("Embedding targets ...")
        tgt_embs, tgt_idx = batched_embed([lp for _, lp in tgt_pairs], preprocess, model, device, emb_cfg.batch_size)
        tgt_pairs = [tgt_pairs[i] for i in tgt_idx]
        if tgt_embs.shape[0] == 0:
            print("[ERROR] target embeddings empty.", file=sys.stderr)
            sys.exit(8)

        print("Embedding others ...")
        oth_embs, oth_idx = batched_embed([lp for _, lp in oth_pairs], preprocess, model, device, emb_cfg.batch_size)
        oth_pairs = [oth_pairs[i] for i in oth_idx]
        if oth_embs.shape[0] == 0:
            print("[ERROR] other embeddings empty.", file=sys.stderr)
            sys.exit(9)

        # Similarities
        print("Computing average top-K similarities ...")
        scores = avg_topk_sim(oth_embs, tgt_embs, emb_cfg.topk)  # numpy [M]
        # sort high->low
        order = np.argsort(-scores)
        scores = scores[order]
        oth_pairs = [oth_pairs[i] for i in order]

        # Sampling by rank bands; per-band k = max(10, 10% of band size)
        print("Sampling review set ...")
        M = len(oth_pairs)
        band_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        random.seed(review_cfg.seed)
        sampled_indices: List[int] = []
        for b in range(5):
            start = int(math.floor(band_edges[b] * M))
            end = int(math.floor(band_edges[b+1] * M))
            bucket = list(range(start, min(end, M)))
            if not bucket:
                continue
            k = max(10, int(round(0.1 * len(bucket))))
            sampled_indices += random.sample(bucket, min(k, len(bucket)))
        sampled_indices = sorted(set(sampled_indices))

        sampled_pairs = [oth_pairs[i] for i in sampled_indices]
        sampled_scores = [float(scores[i]) for i in sampled_indices]
        remaining_pairs = [oth_pairs[i] for i in range(M) if i not in sampled_indices]
        remaining_scores = [float(scores[i]) for i in range(M) if i not in sampled_indices]

        # Stage 1: review sampled
        cv2.namedWindow(display_cfg.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(display_cfg.window_name, display_cfg.default_window_wh[1], display_cfg.default_window_wh[0])

        moved_sim: List[float] = []
        ignored_sim: List[float] = []
        rows_stage1 = []

        print("\nStage 1: sampled review. 'j' move, 'k' ignore, 'esc' abort.")
        for (rp, lp), sim in zip(sampled_pairs, sampled_scores):
            img = cv2.imread(lp)
            if img is None:
                print(f"[WARN] cannot read {lp}")
                continue
            H, W = get_window_size(display_cfg.window_name, display_cfg.default_window_wh)
            vis = scale_to_fit(img, H, W)
            head = f"{rp} | sim={sim:.4f} | SAMPLE"
            vis = draw_top_text(vis, head, display_cfg.font_scale, display_cfg.font_thick)
            cv2.imshow(display_cfg.window_name, vis)
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC
                print("Aborted.")
                cv2.destroyAllWindows()
                ftp.close()
                return
            if key == ord('j'):
                try:
                    new_remote = ftp.move_file(rp, target_dir)
                    moved_sim.append(sim)
                    decision = "move"
                except Exception as e:
                    print(f"[ERROR] move failed {rp}: {e}", file=sys.stderr)
                    new_remote = ""
                    decision = "move_failed"
            else:
                ignored_sim.append(sim)
                new_remote = ""
                decision = "ignore"
            rows_stage1.append((rp, new_remote, sim, decision, "sample"))
        if rows_stage1:
            log_csv_append(log_cfg.csv_path, rows_stage1)

        # Threshold and summary
        if moved_sim or ignored_sim:
            x = moved_sim + ignored_sim
            y = [1]*len(moved_sim) + [0]*len(ignored_sim)
            th = weighted_threshold(x, y)
            lo_moved = min(moved_sim) if moved_sim else float("nan")
            hi_ignored = max(ignored_sim) if ignored_sim else float("nan")
            print("\n--- Threshold summary ---")
            print(f"Lowest similarity MOVED   : {lo_moved if moved_sim else 'n/a'}")
            print(f"Highest similarity IGNORED: {hi_ignored if ignored_sim else 'n/a'}")
            print(f"Weighted threshold        : {th:.6f}")
        else:
            th = float("nan")
            print("\nNo sampled decisions. Threshold unavailable.")

        # Stage 2: remaining high->low
        print("\nStage 2: remaining review. 'j' move, 'k' ignore, 'esc' finish.")
        rows_stage2 = []
        for (rp, lp), sim in zip(remaining_pairs, remaining_scores):
            img = cv2.imread(lp)
            if img is None:
                print(f"[WARN] cannot read {lp}")
                continue
            H, W = get_window_size(display_cfg.window_name, display_cfg.default_window_wh)
            vis = scale_to_fit(img, H, W)
            hint = f"{rp} | sim={sim:.4f}" + (f" | TH={th:.4f}" if math.isfinite(th) else "")
            vis = draw_top_text(vis, hint, display_cfg.font_scale, display_cfg.font_thick)
            cv2.imshow(display_cfg.window_name, vis)
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                print("Done.")
                break
            if key == ord('j'):
                try:
                    new_remote = ftp.move_file(rp, target_dir)
                    decision = "move"
                except Exception as e:
                    print(f"[ERROR] move failed {rp}: {e}", file=sys.stderr)
                    new_remote = ""
                    decision = "move_failed"
            else:
                new_remote = ""
                decision = "ignore"
            rows_stage2.append((rp, new_remote, sim, decision, "remaining"))
        if rows_stage2:
            log_csv_append(log_cfg.csv_path, rows_stage2)

        cv2.destroyAllWindows()
        ftp.close()
        print(f"\nLog: {log_cfg.csv_path}")

    finally:
        try:
            ftp.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
