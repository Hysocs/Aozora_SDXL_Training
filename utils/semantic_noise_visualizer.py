import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import torch

# --- Core Logic Functions (Copied VERBATIM from training_code.py) ---

def generate_character_map(pil_image):
    """Generates a map focusing on the overall character/object region using color and structure."""
    np_image_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    bilateral = cv2.bilateralFilter(np_image_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    lab_image = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    mean_a, mean_b = np.mean(a_channel), np.mean(b_channel)
    saliency_a = np.abs(a_channel.astype(np.float32) - mean_a)
    saliency_b = np.abs(b_channel.astype(np.float32) - mean_b)
    color_saliency = saliency_a + saliency_b
    if color_saliency.max() > 0:
        color_saliency = (color_saliency / color_saliency.max() * 255).astype(np.uint8)
    kernel = np.ones((11, 11), np.uint8)
    dilated = cv2.dilate(color_saliency, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=2)
    final_map = cv2.GaussianBlur(eroded, (11, 11), 0)
    return Image.fromarray(final_map)

def generate_detail_map(pil_image):
    """Generates a map focusing on edges and lineart."""
    np_image_gray = np.array(pil_image.convert("L"))
    sobelx = cv2.Sobel(np_image_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(np_image_gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    if magnitude.max() > 0:
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
    else:
        magnitude = np.zeros_like(magnitude, dtype=np.uint8)
    final_map = cv2.GaussianBlur(magnitude, (1, 1), 0)
    return Image.fromarray(final_map)

def generate_semantic_noise_visualization(combined_map_pil, strength=1.0, high_contrast=False):
    """Generates a visual representation of the semantic noise from the combined map."""
    width, height = combined_map_pil.size
    noise = torch.randn(1, 3, height, width)
    salience_tensor = torch.from_numpy(np.array(combined_map_pil.convert("L"))).float() / 255.0
    salience_tensor = salience_tensor.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)
    modulation = 1.0 + strength * salience_tensor
    structured_noise = noise * modulation
    std_structured = torch.std(structured_noise, dim=(2,3), keepdim=True)
    std_original = torch.std(noise, dim=(2,3), keepdim=True)
    final_noise = structured_noise * (std_original / (std_structured + 1e-9))
    if high_contrast:
        # For visualization, just show the map itself. It's more intuitive.
        return combined_map_pil
    else:
        display_noise = (final_noise + 2.5) / 5.0
        display_noise = torch.clamp(display_noise, 0, 1)
        noise_np = display_noise.squeeze(0).permute(1, 2, 0).numpy()
        return Image.fromarray((noise_np * 255).astype(np.uint8))


# --- GUI Application Class ---

class SimpleVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Semantic Noise Simulator (Training Replica)")
        self.geometry("1200x600")
        self.minsize(900, 500)

        # --- Stored data ---
        self.original_image = None
        self.char_map_np = None # Store numpy arrays for speed
        self.detail_map_np = None
        self.display_size = (300, 300)

        # --- Style ---
        style = ttk.Style(self)
        style.theme_use('clam')
        # ... (your styling can be pasted here if you like)

        # --- Main Layout ---
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # --- Top Controls ---
        top_frame = ttk.Frame(self, padding=10)
        top_frame.grid(row=0, column=0, sticky="ew")
        self.btn_load = ttk.Button(top_frame, text="Load Image", command=self.load_image)
        self.btn_load.pack(side=tk.LEFT)

        # --- Image Display Area ---
        images_frame = ttk.Frame(self, padding=10)
        images_frame.grid(row=1, column=0, sticky="nsew")
        for i in range(3):
            images_frame.columnconfigure(i, weight=1, uniform="group1")
        images_frame.rowconfigure(0, weight=1)

        self.canvas_original = self.create_image_panel(images_frame, "Original Image", 0)
        self.canvas_combined = self.create_image_panel(images_frame, "Final Combined Map", 1)
        self.canvas_noise = self.create_image_panel(images_frame, "Noise Visualization", 2)

        # --- Bottom Controls ---
        controls_frame = ttk.LabelFrame(self, text="Training Parameters", padding=10)
        controls_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        
        self.char_weight_var = self.create_slider(controls_frame, "Character Weight", 0.0, 5.0, 1.0)
        self.detail_weight_var = self.create_slider(controls_frame, "Detail Weight", 0.0, 5.0, 0.5)
        
        # --- Checkboxes ---
        checkbox_frame = ttk.Frame(controls_frame)
        checkbox_frame.pack(fill='x', expand=True, pady=5)
        
        self.normalize_var = tk.BooleanVar(value=False) # Default to 'clipping/stacking'
        chk_normalize = ttk.Checkbutton(checkbox_frame, text="Normalize (Rescale)", variable=self.normalize_var, command=self.update_display)
        chk_normalize.pack(side=tk.LEFT, padx=10)
        
        self.high_contrast_var = tk.BooleanVar(value=True) # Default to showing the map
        chk_contrast = ttk.Checkbutton(checkbox_frame, text="High Contrast Noise Viz", variable=self.high_contrast_var, command=self.update_display)
        chk_contrast.pack(side=tk.LEFT, padx=10)

    def create_image_panel(self, parent, title, col_index):
        frame = ttk.LabelFrame(parent, text=title)
        frame.grid(row=0, column=col_index, sticky="nsew", padx=5, pady=5)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        canvas = tk.Canvas(frame, bg="#333")
        canvas.grid(sticky="nsew")
        return canvas

    def create_slider(self, parent, label_text, min_val, max_val, default):
        container = ttk.Frame(parent)
        container.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(container, text=label_text, width=15).pack(side=tk.LEFT)
        
        var = tk.DoubleVar(value=default)
        
        label_val = ttk.Label(container, text=f"{var.get():.2f}", width=5)
        label_val.pack(side=tk.RIGHT)
        
        scale = ttk.Scale(container, from_=min_val, to=max_val, orient=tk.HORIZONTAL, variable=var,
                          command=lambda val, v=var, l=label_val: self.on_slider_change(v, l))
        scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        return var
        
    def on_slider_change(self, var, label):
        label.config(text=f"{var.get():.2f}")
        self.update_display()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp")])
        if not file_path:
            return
        
        self.original_image = Image.open(file_path).convert("RGB")
        
        # --- Pre-calculate base maps ONCE for efficiency ---
        print("Generating base maps for new image...")
        self.char_map_np = np.array(generate_character_map(self.original_image)).astype(np.float32)
        self.detail_map_np = np.array(generate_detail_map(self.original_image)).astype(np.float32)
        print("Base maps generated.")

        self.display_on_canvas(self.canvas_original, self.original_image)
        self.update_display()
        
    def update_display(self):
        if self.original_image is None:
            return

        # 1. Get weights from sliders
        char_weight = self.char_weight_var.get()
        detail_weight = self.detail_weight_var.get()

        # 2. Combine the pre-calculated numpy maps
        combined_map_np = (char_weight * self.char_map_np) + (detail_weight * self.detail_map_np)

        # 3. Apply normalization strategy based on the checkbox
        if self.normalize_var.get():
            # RESCALE: Find the max value and scale everything down
            if combined_map_np.max() > 0:
                combined_map_np = (combined_map_np / combined_map_np.max() * 255)
        else:
            # CLIP: Just cap values at 255 (the "stacking" behavior)
            combined_map_np = np.clip(combined_map_np, 0, 255)

        # 4. Convert to PIL image for display and final noise generation
        combined_map_pil = Image.fromarray(combined_map_np.astype(np.uint8))
        
        # 5. Generate noise visualization
        noise_map_pil = generate_semantic_noise_visualization(
            combined_map_pil,
            strength=5.0, # Strength is for visualization only, can be hardcoded
            high_contrast=self.high_contrast_var.get()
        )

        # 6. Update the canvases
        self.display_on_canvas(self.canvas_combined, combined_map_pil)
        self.display_on_canvas(self.canvas_noise, noise_map_pil)
        
    def display_on_canvas(self, canvas, pil_image):
        if pil_image is None:
            return
            
        canvas_w, canvas_h = canvas.winfo_width(), canvas.winfo_height()
        if canvas_w < 2 or canvas_h < 2: # Canvas not ready yet
            self.after(50, lambda: self.display_on_canvas(canvas, pil_image))
            return
            
        img_w, img_h = pil_image.size
        aspect = img_w / img_h
        
        if canvas_w / canvas_h > aspect:
            new_h = canvas_h
            new_w = int(new_h * aspect)
        else:
            new_w = canvas_w
            new_h = int(new_w / aspect)
            
        img_resized = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)
        
        canvas.delete("all")
        canvas.create_image(canvas_w/2, canvas_h/2, anchor=tk.CENTER, image=img_tk)
        canvas.image = img_tk # Keep a reference!


if __name__ == "__main__":
    app = SimpleVisualizer()
    app.mainloop()