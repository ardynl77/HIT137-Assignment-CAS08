print("="*30)
print("""
Ardyn Low - S395694
Theo Rothmann - S366484
""")
print("="*30)

print("""
Assignment 3""")
print()

"""
Spot the Difference — simplified single-file desktop game.
Run:  python3 spot_the_difference.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import random
from abc import ABC, abstractmethod


# ── Alteration base + subclasses (inheritance / polymorphism) ─────────────────

class Alteration(ABC):    
    @abstractmethod
    def apply(self, img, x, y, w, h, mask=None):
        pass

    def apply_mask(self, img, x, y, w, h, changed, mask=None):
        roi = img[y:y+h, x:x+w]

        if mask is None:
            img[y:y+h, x:x+w] = changed
        else:
            soft_mask = cv2.GaussianBlur(mask, (45,45), 0)
            soft_mask = soft_mask.astype("float32") / 255.0
            soft_mask = soft_mask[:, :, None]
        
            strength = 1.0
            soft_mask = soft_mask * strength
            

            blended = (changed * soft_mask + roi * (1 - soft_mask)).astype("uint8")
            img[y:y+h, x:x+w] = blended

class ColourShift(Alteration):
    def apply(self, img, x, y, w, h, mask=None):
        roi = img[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype("int16")
        hsv[:,:,0] = (hsv[:,:,0] + 60) % 180
        changed = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
        self.apply_mask(img, x, y, w, h, changed, mask)

class Blur(Alteration):
    def apply(self, img, x, y, w, h, mask=None):
        roi = img[y:y+h, x:x+w]

        changed = cv2.GaussianBlur(roi, (45, 45), 0)
        self.apply_mask(img, x, y, w, h, changed, mask)

class Invert(Alteration):
    def apply(self, img, x, y, w, h, mask=None):
        roi = img[y:y+h, x:x+w]

        changed = cv2.bitwise_not(roi)
        self.apply_mask(img, x, y, w, h, changed, mask)

class Darken(Alteration):
    def apply(self, img, x, y, w, h, mask=None):
        roi = img[y:y+h, x:x+w].astype("int16")
        changed = np.clip(roi - 80, 0, 255).astype("uint8")
        self.apply_mask(img, x, y, w, h, changed, mask)

class EdgeDetect(Alteration):
    def apply(self, img, x, y, w, h, mask=None):
        roi = img[y:y+h, x:x+w]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        changed = roi.copy()
        changed[edges > 0] = [0, 0, 255]

        self.apply_mask(img, x, y, w, h, changed, mask)

ALTERATIONS = [ColourShift, Blur, Invert, Darken,EdgeDetect]




# ── GameImage: load + create modified clone with 5 differences ────────────────

class GameImage:
    NUM_DIFFS = 5
    def __init__(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError("Cannot open image.")
        h, w = img.shape[:2]
        if max(h, w) > 500:
            s = 500 / max(h, w)
            img = cv2.resize(img, (int(w*s), int(h*s)))
        self.original = img
        self.modified = img.copy()
        self.regions  = []          # list of (x, y, w, h)
        self._place_differences()


    def _place_differences(self):
        ih, iw = self.original.shape[:2]
        placed = []
        alt_types = random.sample(ALTERATIONS, self.NUM_DIFFS)
        taken_mask = np.zeros((ih, iw), dtype=np.uint8)
        def make_random_mask(w, h):
            mask = np.zeros((h, w), dtype=np.uint8)

            shape = random.choice(["circle", "ellipse", "blob", "triangle"])

            if shape == "circle":
                r = random.randint(min(w, h) // 4, min(w, h) // 3)
                cv2.circle(mask, (w//2, h//2), r, 255, -1)

            elif shape == "ellipse":
                center = (w // 2, h // 2)
                axes = (
                    random.randint(w // 5, w // 3),
                    random.randint(h // 5, h // 3)
                )
                angle = random.randint(0, 180)
                cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)

            elif shape == "triangle":
                pad = 4
                pts = np.array([
                    [random.randint(pad, w-pad), random.randint(pad, h-pad)],
                    [random.randint(pad, w-pad), random.randint(pad, h-pad)],
                    [random.randint(pad, w-pad), random.randint(pad, h-pad)]
                ], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)

            else:  # blob
                cx, cy = w // 2, h // 2
                points = []

                for i in range(10):
                    angle = 2 * np.pi * i / 10
                    rx = random.randint(w // 5, w // 3)
                    ry = random.randint(h // 5, h // 3)

                    px = int(cx + np.cos(angle) * rx)
                    py = int(cy + np.sin(angle) * ry)
                    points.append([px, py])

                points = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)

            return mask


        for alt_cls in alt_types:
            rw = random.randint(60, 120)
            rh = random.randint(60, 120)
            mask = make_random_mask(rw, rh)

            for _ in range(500):
                max_rw = min(120, iw - 20)
                max_rh = min(120, ih - 20)

                if max_rw < 60 or max_rh < 60:
                    raise ValueError("Image is too small. Try a larger image.")

                rw = random.randint(60, max_rw)
                rh = random.randint(60, max_rh)

                padded_mask = cv2.dilate(mask, np.ones((30, 30), np.uint8), iterations=1)

                area_taken = taken_mask[y:y+rh, x:x+rw]
                overlap = np.any((area_taken == 255) & (padded_mask == 255))

                if not overlap:
                    before = self.modified[y:y+rh, x:x+rw].copy()
                    alt_cls().apply(self.modified, x, y, rw, rh, mask)
                    after = self.modified[y:y+rh, x:x+rw]

                    diff_score = np.mean(cv2.absdiff(before, after))

                    if diff_score < 12:
                        self.modified[y:y+rh, x:x+rw] = before
                        continue

                    area_taken[padded_mask == 255] = 255

                    placed.append((x, y, rw, rh))
                    break

        self.regions = placed
        if len(self.regions) < self.NUM_DIFFS:
            raise ValueError("Could not place all 5 differences. Try a larger image.")


# ── GameState: click validation, mistakes, found tracking ─────────────────────

class GameState:
    MAX_MISTAKES = 3
    HIT_PAD      = 25

    def __init__(self, game_image):
        self.gi        = game_image
        self.found     = [False] * game_image.NUM_DIFFS
        self.mistakes  = 0

    @property
    def remaining(self):
        return self.found.count(False)

    @property
    def locked(self):
        return self.mistakes >= self.MAX_MISTAKES

    def click(self, cx, cy):
        """Return index of hit region, or -1 on miss."""
        if self.locked or self.remaining == 0:
            return -1
        for i, (x, y, w, h) in enumerate(self.gi.regions):
            if not self.found[i]:
                p = self.HIT_PAD
                if x-p <= cx <= x+w+p and y-p <= cy <= y+h+p:
                    self.found[i] = True
                    return i
        self.mistakes += 1
        return -1


# ========== Application: Tkinter GUI ====================================

class Application:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Spot the 5 Differences")
        self.root.resizable(False, False)

        self.gi    = None
        self.gs    = None
        self._refs = {}     # keep PhotoImage references alive
        self.total_score = 0
        self._build_ui()
        self.root.mainloop()
        

    def _build_ui(self):
        # ========== top bar ====================================
        bar = tk.Frame(self.root, bg="#222")
        bar.pack(fill=tk.X)

        tk.Button(bar, text="Load Image", command=self._load,
                  bg="#0055aa", fg="white", padx=10).pack(side=tk.LEFT, padx=6, pady=6)
        tk.Button(bar, text="Reveal All", command=self._reveal,
                  bg="#885500", fg="white", padx=10).pack(side=tk.LEFT, padx=2, pady=6)

        self.lbl_remaining = tk.Label(bar, text="Remaining: —",
                                      bg="#222", fg="#00ee66", font=("Courier", 12, "bold"))
        self.lbl_remaining.pack(side=tk.LEFT, padx=20)

        self.lbl_mistakes = tk.Label(bar, text="Mistakes: 0/3",
                                     bg="#222", fg="white", font=("Courier", 12, "bold"))
        self.lbl_mistakes.pack(side=tk.LEFT)

        self.lbl_score = tk.Label(
            bar,
            text="Score: 0",
            bg="#222",
            fg="#ffaa00",
            font=("Courier", 12, "bold")
        )
        self.lbl_score.pack(side=tk.LEFT, padx=20)


        # ── canvases ──────────────────────────────────────────────────────────
        frame = tk.Frame(self.root)
        frame.pack()

        tk.Label(frame, text="Original").grid(row=0, column=0)
        tk.Label(frame, text="Modified  ← click here").grid(row=0, column=1)

        self.can_orig = tk.Canvas(frame, bg="#111", width=400, height=300)
        self.can_orig.grid(row=1, column=0, padx=4, pady=4)

        self.can_mod  = tk.Canvas(frame, bg="#111", width=400, height=300,
                                  cursor="crosshair")
        self.can_mod.grid(row=1, column=1, padx=4, pady=4)
        self.can_mod.bind("<Button-1>", self._on_click)

        # ── status ────────────────────────────────────────────────────────────
        self.lbl_status = tk.Label(self.root, text="Load an image to start.",
                                   fg="#888", font=("Helvetica", 10))
        self.lbl_status.pack(pady=4)

    # load function

    def _load(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")])
        if not path:
            return
        try:
            self.gi = GameImage(path)
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        self.gs           = GameState(self.gi)
        self.orig_display = self.gi.original.copy()
        self.mod_display  = self.gi.modified.copy()

        self._show(self.can_orig, self.orig_display, "orig")
        self._show(self.can_mod,  self.mod_display,  "mod")
        self._refresh()
        self.lbl_status.config(text="Find the 5 differences on the right image.")

    # click function

    def _on_click(self, event):
        if self.gs is None:
            return
        if self.gs.locked:
            self.lbl_status.config(text="Locked out — load a new image.")
            return
        if self.gs.remaining == 0:
            self.lbl_status.config(text="All differences already found — load a new image.")
            return
        idx = self.gs.click(event.x, event.y)

        if idx >= 0:
            
            self.total_score += 1
            x, y, w, h = self.gi.regions[idx]
            cx, cy, r  = x + w//2, y + h//2, max(w, h)//2 + 10
            self._circle(self.orig_display, cx, cy, r, (0, 0, 220))
            self._circle(self.mod_display,  cx, cy, r, (0, 0, 220))
            self._show(self.can_orig, self.orig_display, "orig")
            self._show(self.can_mod,  self.mod_display,  "mod")
            self._refresh()
            
            
            if self.gs.remaining == 0:
                messagebox.showinfo("Done!", "All 5 found! Load another image.")
        else:
            self.total_score = max(0, self.total_score - 1)
            self._refresh()
            if self.gs.locked:
                self.lbl_status.config(text="3 mistakes — locked out!")
                messagebox.showwarning("Locked Out",
                    f"Too many mistakes!\nFound: {self.gs.found.count(True)}/5\n"
                    "Press Reveal All or load a new image.")
            else:
                left = self.gs.MAX_MISTAKES - self.gs.mistakes
                self.lbl_status.config(text=f"Wrong! {left} mistake(s) left.")

    # reveal function

    def _reveal(self):
        if self.gs is None:
            return
        for i, (x, y, w, h) in enumerate(self.gi.regions):
            if not self.gs.found[i]:
                cx, cy, r = x + w//2, y + h//2, max(w, h)//2 + 10
                self._circle(self.orig_display, cx, cy, r, (200, 100, 0))
                self._circle(self.mod_display,  cx, cy, r, (200, 100, 0))
        self._show(self.can_orig, self.orig_display, "orig")
        self._show(self.can_mod,  self.mod_display,  "mod")
        self._refresh()
        self.lbl_status.config(text="Differences revealed in blue. Load a new image.")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _circle(self, img, cx, cy, r, bgr):
        cv2.circle(img, (cx, cy), r, bgr, 3)

    def _show(self, canvas, bgr_img, key):
        h, w = bgr_img.shape[:2]
        canvas.config(width=w, height=h)
        rgb    = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._refs[key] = tk_img
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=tk_img)

    def _refresh(self):
        if self.gs is None:
            return
        self.lbl_remaining.config(text=f"Remaining: {self.gs.remaining}")
        self.lbl_mistakes.config(
            text=f"Mistakes: {self.gs.mistakes}/{self.gs.MAX_MISTAKES}",
            fg="#ff4444" if self.gs.locked else "white")
        self.lbl_score.config(text=f"Score: {self.total_score}")


if __name__ == "__main__":
    Application()