"""Unified illustration-detail detector and standalone preview GUI."""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def generate_illustration_detail_map(pil_image: Image.Image, sensitivity: float = 0.55) -> np.ndarray:
    """Return a fast [H,W] map of illustration lines and fine texture.

    Lines and texture are both local high-frequency changes, so one Laplacian
    response detects them together. A small neighbourhood average favors useful
    coherent detail over isolated pixel noise.
    """
    rgb = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (3, 3), 0.55)
    detail = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))

    # Coherence gently raises clustered detail without erasing a clean contour.
    coherence = cv2.blur(detail, (5, 5))
    coherence /= max(float(np.percentile(coherence, 99.0)), 1.0e-6)
    detail *= 0.65 + 0.35 * np.clip(coherence, 0.0, 1.0)

    # One robust scaling pass. Higher sensitivity lowers the acceptance floor.
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    floor = float(np.percentile(detail, 88.0 - sensitivity * 48.0))
    ceiling = float(np.percentile(detail, 99.5))
    detail = np.clip((detail - floor) / max(ceiling - floor, 1.0e-6), 0.0, 1.0)
    return np.clip(detail, 0.0, 1.0).astype(np.float32)


def generate_lineart_loss_map(
    pil_image: Image.Image,
    latent_h: int,
    latent_w: int,
    oversample: int = 4,
) -> torch.Tensor:
    """Compatibility entry point used by the quantized repair trainer."""
    detail = generate_illustration_detail_map(pil_image, sensitivity=0.55)
    oversample = max(1, int(oversample))
    resized = cv2.resize(
        detail,
        (int(latent_w) * oversample, int(latent_h) * oversample),
        interpolation=cv2.INTER_AREA,
    )
    return torch.from_numpy(resized).unsqueeze(0).to(dtype=torch.float16).contiguous()


def _run_semantic_preview_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import ImageTk

    class SemanticPreviewApp:
        def __init__(self, root):
            self.root = root
            self.root.title("Aozora Semantic Detail Preview")
            self.root.geometry("1480x760")
            self.source_image = None
            self.source_path = None
            self.preview_refs = []
            self.pending_update = None

            toolbar = ttk.Frame(root, padding=8)
            toolbar.pack(fill="x")
            ttk.Button(toolbar, text="Load image", command=self.load_image).pack(side="left")
            ttk.Button(toolbar, text="Save map", command=self.save_map).pack(side="left", padx=(8, 18))

            self.sensitivity_var = tk.DoubleVar(value=0.55)
            self.overlay_var = tk.DoubleVar(value=0.62)
            self._add_slider(toolbar, "Detail sensitivity", self.sensitivity_var)
            self._add_slider(toolbar, "Overlay", self.overlay_var)

            self.status = tk.StringVar(value="Load an image to inspect its line and detail importance map.")
            ttk.Label(root, textvariable=self.status, padding=(10, 0, 10, 8)).pack(fill="x")

            previews = ttk.Frame(root, padding=(8, 0, 8, 8))
            previews.pack(fill="both", expand=True)
            self.panels = []
            for title in ("Original", "Semantic importance", "Overlay"):
                frame = ttk.LabelFrame(previews, text=title, padding=5)
                frame.pack(side="left", fill="both", expand=True, padx=4)
                label = ttk.Label(frame, anchor="center")
                label.pack(fill="both", expand=True)
                label.bind("<Configure>", lambda _event: self.schedule_update())
                self.panels.append(label)

        def _add_slider(self, parent, title, variable):
            frame = ttk.Frame(parent)
            frame.pack(side="left", padx=8)
            ttk.Label(frame, text=title).pack(anchor="w")
            ttk.Scale(frame, variable=variable, from_=0.0, to=1.0, length=180,
                      command=lambda _value: self.schedule_update()).pack(side="left")
            value_label = ttk.Label(frame, width=5)
            value_label.pack(side="left", padx=(4, 0))

            def refresh(*_args):
                value_label.configure(text=f"{variable.get():.2f}")
            variable.trace_add("write", refresh)
            refresh()

        def load_image(self):
            path = filedialog.askopenfilename(
                title="Select an image",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"), ("All files", "*.*")],
            )
            if not path:
                return
            try:
                with Image.open(path) as loaded:
                    self.source_image = loaded.convert("RGB").copy()
                self.source_path = Path(path)
                self.update_previews()
            except Exception as exc:
                messagebox.showerror("Could not load image", str(exc))

        def make_map(self):
            return generate_illustration_detail_map(self.source_image, self.sensitivity_var.get())

        def schedule_update(self):
            if self.source_image is None:
                return
            if self.pending_update is not None:
                self.root.after_cancel(self.pending_update)
            self.pending_update = self.root.after(100, self.update_previews)

        @staticmethod
        def heatmap_from_map(detail_map):
            map_u8 = np.clip(detail_map * 255.0, 0, 255).astype(np.uint8)
            heat_bgr = cv2.applyColorMap(map_u8, cv2.COLORMAP_TURBO)
            return Image.fromarray(cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB))

        @staticmethod
        def fit_panel(image, label):
            fitted = image.copy()
            fitted.thumbnail(
                (max(200, label.winfo_width() - 10), max(200, label.winfo_height() - 10)),
                Image.Resampling.LANCZOS,
            )
            return fitted

        def update_previews(self):
            self.pending_update = None
            if self.source_image is None:
                return
            try:
                detail_map = self.make_map()
                heat = self.heatmap_from_map(detail_map)
                overlay = Image.blend(self.source_image, heat, float(self.overlay_var.get()))
                self.preview_refs = []
                for label, preview in zip(self.panels, (self.source_image, heat, overlay)):
                    photo = ImageTk.PhotoImage(self.fit_panel(preview, label))
                    label.configure(image=photo)
                    self.preview_refs.append(photo)
                selected = float((detail_map >= 0.5).mean() * 100.0)
                self.status.set(
                    f"{self.source_path.name} — {self.source_image.width}x{self.source_image.height} — "
                    f"{selected:.1f}% above 50% importance"
                )
            except Exception as exc:
                self.status.set(f"Preview failed: {exc}")

        def save_map(self):
            if self.source_image is None:
                messagebox.showinfo("No image", "Load an image first.")
                return
            path = filedialog.asksaveasfilename(
                title="Save semantic map",
                defaultextension=".png",
                initialfile=f"{self.source_path.stem}_semantic.png",
                filetypes=[("PNG image", "*.png")],
            )
            if path:
                Image.fromarray(np.clip(self.make_map() * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)
                self.status.set(f"Saved semantic map: {path}")

    root = tk.Tk()
    SemanticPreviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    _run_semantic_preview_gui()
