import os
import cv2
import csv
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# ---------------- Config ----------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
WINDOW_NAME = "Image Categorizer"
LOG_FILE = "move_log.csv"

# ---------------- Helpers ----------------
def select_folders(prompt):
    folders = []
    while True:
        folder = filedialog.askdirectory(title=prompt)
        if not folder:
            break
        folders.append(Path(folder))
    return folders

def get_all_images(scan_folders, exclude_folders):
    files = []
    exclude_set = {str(f.resolve()) for f in exclude_folders}
    for folder in scan_folders:
        for root, _, filenames in os.walk(folder):
            if any(str(root).startswith(ex) for ex in exclude_set):
                continue
            for name in filenames:
                if Path(name).suffix.lower() in IMAGE_EXTS:
                    files.append(Path(root) / name)
    return files

def fit_to_window(image, window_size):
    h, w = image.shape[:2]
    target_h, target_w = window_size
    scale = min(target_w / w, target_h / h)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

# ---------------- Main ----------------
def main():
    root = tk.Tk()
    root.withdraw()

    print("Select scan folders (Cancel to finish):")
    scan_folders = select_folders("Select Scan Folder")
    print("Select category folders (Cancel to finish, less than 9):")
    category_folders = select_folders("Select Category Folder")[:9]

    if not scan_folders or not category_folders:
        print("No folders selected. Exiting.")
        return

    img_files = get_all_images(scan_folders, category_folders)
    if not img_files:
        print("No images found.")
        return

    log_rows = []

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 700, 1200)

    for img_path in img_files:
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        # Fit image to current window
        h = cv2.getWindowImageRect(WINDOW_NAME)[3]
        w = cv2.getWindowImageRect(WINDOW_NAME)[2]
        img_resized = fit_to_window(img, (h, w))

        display = img_resized.copy()

        # Add top text (path)
        cv2.rectangle(display, (0, 0), (display.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(display, str(img_path), (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add bottom text (categories)
        h2 = display.shape[0]
        cv2.rectangle(display, (0, h2 - 80), (display.shape[1], h2), (0, 0, 0), -1)
        for i, f in enumerate(category_folders, start=1):
            cv2.putText(display, f"[{i}] {f}", (10, h2 - 60 + (i - 1) * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(display, "[0] Skip", (10, h2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(0)

        if key == 27:  # ESC
            break

        new_path = ""
        if ord('1') <= key <= ord(str(len(category_folders))):
            idx = int(chr(key)) - 1
            dest = category_folders[idx] / img_path.name
            base, ext = os.path.splitext(dest)
            counter = 1
            while dest.exists():
                dest = Path(f"{base}_{counter}{ext}")
                counter += 1
            os.rename(img_path, dest)
            new_path = str(dest)
        else:
            new_path = ""

        log_rows.append([str(img_path), new_path])

    cv2.destroyAllWindows()

    with open(LOG_FILE, "w", newline='', encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(log_rows)

    print(f"Done. Log saved to {LOG_FILE}")

if __name__ == "__main__":
    import numpy as np
    main()
