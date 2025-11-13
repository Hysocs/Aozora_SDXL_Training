import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import torch
from torchvision.transforms import functional as F

# --- Core Logic Functions (Unchanged from training code) ---
def generate_character_map(pil_image):
    np_image_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    bilateral = cv2.bilateralFilter(np_image_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    lab_image = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    mean_a, mean_b = np.mean(a_channel), np.mean(b_channel)
    saliency_a = np.abs(a_channel.astype(np.float32) - mean_a)
    saliency_b = np.abs(b_channel.astype(np.float32) - mean_b)
    color_saliency = saliency_a + saliency_b
    if color_saliency.max() > 0:
        color_saliency_norm = color_saliency / color_saliency.max()
    else:
        color_saliency_norm = np.zeros_like(color_saliency, dtype=np.float32)
    color_saliency_uint8 = (color_saliency_norm * 255).astype(np.uint8)
    kernel = np.ones((11, 11), np.uint8)
    dilated = cv2.dilate(color_saliency_uint8, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=2)
    eroded_float = eroded.astype(np.float32) / 255.0
    final_map = cv2.GaussianBlur(eroded_float, (11, 11), 0)
    return Image.fromarray(final_map)

def generate_detail_map(pil_image):
    np_image_gray = np.array(pil_image.convert("L"))
    sobelx = cv2.Sobel(np_image_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(np_image_gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    if magnitude.max() > 0:
        magnitude_norm = magnitude / magnitude.max()
    else:
        magnitude_norm = np.zeros_like(magnitude, dtype=np.float32)
    final_map = cv2.GaussianBlur(magnitude_norm.astype(np.float32), (1, 1), 0)
    return Image.fromarray(final_map)

# --- FIXED Visualization Function ---
def generate_modulation_heatmap(weight_map_01, global_strength):
    """
    FIXED: Now properly visualizes effective modulation (1.0 to 1.0+strength)
    using the full color range regardless of weight_map values.
    """
    # Calculate effective modulation
    effective_modulation = 1.0 + weight_map_01 * global_strength
    
    # Scale to 0-255 using ACTUAL min/max to always use full color range
    min_mod = effective_modulation.min()
    max_mod = effective_modulation.max()
    
    if max_mod > min_mod:
        # Normalize to 0-255 mapping min->blue, max->red
        scaled = ((effective_modulation - min_mod) / (max_mod - min_mod)) * 255
    else:
        # Uniform map - all same color
        scaled = np.zeros_like(effective_modulation) * 255
    
    heatmap_bgr = cv2.applyColorMap(scaled.astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(heatmap_rgb), min_mod, max_mod

def generate_blocky_noise(weight_map_01, global_strength, block_size=16):
    """Blocky noise visualization (unchanged logic)"""
    width, height = weight_map_01.shape[1], weight_map_01.shape[0]
    block_size = max(1, int(block_size))
    small_w = width // block_size
    small_h = height // block_size
    if small_w < 1 or small_h < 1: 
        return Image.fromarray((weight_map_01 * 255).astype(np.uint8))
    
    base_noise_small = torch.randn(1, 3, small_h, small_w)
    salience_tensor_small = torch.from_numpy(weight_map_01).float().unsqueeze(0).unsqueeze(0)
    salience_tensor_small = F.resize(salience_tensor_small, (small_h, small_w), 
                                     interpolation='bilinear', antialias=True).squeeze().numpy()
    salience_tensor_small = salience_tensor_small.reshape(1, 1, small_h, small_w).expand_as(base_noise_small)
    
    modulation = 1.0 + salience_tensor_small * global_strength
    structured_noise_small = base_noise_small * modulation
    
    display_noise = torch.clamp((structured_noise_small + 4) / 8.0, 0, 1)
    display_noise_np = display_noise.squeeze(0).permute(1, 2, 0).numpy()
    noise_img_small = Image.fromarray((display_noise_np * 255).astype(np.uint8))
    noise_img_large = noise_img_small.resize((width, height), Image.Resampling.NEAREST)
    
    return noise_img_large

# --- GUI Application Class ---
class SimpleVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Semantic Noise Simulator (Training Replica)")
        self.geometry("1350x780")
        self.minsize(1000, 500)

        self.original_image = None
        self.char_map_np = None
        self.detail_map_np = None

        style = ttk.Style(self)
        style.theme_use('clam')

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top_frame = ttk.Frame(self, padding=10)
        top_frame.grid(row=0, column=0, sticky="ew")
        self.btn_load = ttk.Button(top_frame, text="Load Image", command=self.load_image)
        self.btn_load.pack(side=tk.LEFT)

        images_frame = ttk.Frame(self, padding=10)
        images_frame.grid(row=1, column=0, sticky="nsew")
        for i in range(3):
            images_frame.columnconfigure(i, weight=1, uniform="group1")
        images_frame.rowconfigure(0, weight=1)

        self.canvas_original = self.create_image_panel(images_frame, "Original Image", 0)
        self.canvas_weight = self.create_image_panel(images_frame, "Normalized Weight Map\n(0.0 to 1.0)", 1)
        self.canvas_noise = self.create_image_panel(images_frame, "Effective Modulation\n(Color shows relative strength)", 2)

        controls_frame = ttk.LabelFrame(self, text="Training Parameters", padding=10)
        controls_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        
        self.blend_var = self.create_slider(controls_frame, "Detail vs Character Balance", 0.0, 1.0, 0.5, is_float=True)
        self.global_strength_var = self.create_slider(controls_frame, "Overall Strength", 0.0, 5.0, 0.8, is_float=True)
        self.block_size_var = self.create_slider(controls_frame, "Noise Block Size", 2, 64, 16, is_float=False)

        stats_frame = ttk.Frame(controls_frame)
        stats_frame.pack(fill='x', expand=True, pady=5)
        ttk.Label(stats_frame, text="Weight Map Range:", font=('TkDefaultFont', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        self.stats_label = ttk.Label(stats_frame, text="[0.00, 0.00]")
        self.stats_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(stats_frame, text="Modulation Range:", font=('TkDefaultFont', 9, 'bold')).pack(side=tk.LEFT, padx=15)
        self.modulation_label = ttk.Label(stats_frame, text="[1.00, 1.00]")
        self.modulation_label.pack(side=tk.LEFT, padx=5)

        checkbox_frame = ttk.Frame(controls_frame)
        checkbox_frame.pack(fill='x', expand=True, pady=5)
        
        self.normalize_var = tk.BooleanVar(value=True)
        chk_normalize = ttk.Checkbutton(checkbox_frame, text="Normalize Weight Map", variable=self.normalize_var, command=self.update_display)
        chk_normalize.pack(side=tk.LEFT, padx=10)
        
        self.show_heat_map_var = tk.BooleanVar(value=True)
        chk_heatmap = ttk.Checkbutton(checkbox_frame, text="Show Heat Map", variable=self.show_heat_map_var, command=self.update_display)
        chk_heatmap.pack(side=tk.LEFT, padx=10)

    def create_image_panel(self, parent, title, col_index):
        frame = ttk.LabelFrame(parent, text=title)
        frame.grid(row=0, column=col_index, sticky="nsew", padx=5, pady=5)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        canvas = tk.Canvas(frame, bg="#333", highlightthickness=0)
        canvas.grid(sticky="nsew")
        return canvas

    def create_slider(self, parent, label_text, min_val, max_val, default, is_float=True):
        container = ttk.Frame(parent)
        container.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(container, text=label_text, width=25).pack(side=tk.LEFT)
        var = tk.DoubleVar(value=default) if is_float else tk.IntVar(value=default)
        val_format = "{:.2f}" if is_float else "{:d}"
        label_val = ttk.Label(container, text=val_format.format(var.get()), width=5)
        label_val.pack(side=tk.RIGHT)
        scale = ttk.Scale(container, from_=min_val, to=max_val, orient=tk.HORIZONTAL, variable=var,
                          command=lambda val, v=var, l=label_val, f=val_format: self.on_slider_change(v, l, f))
        scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        return var
        
    def on_slider_change(self, var, label, val_format):
        label.config(text=val_format.format(var.get()))
        self.update_display()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp")])
        if not file_path: return
        self.original_image = Image.open(file_path).convert("RGB")
        print("Generating base maps for new image...")
        self.char_map_np = np.array(generate_character_map(self.original_image)).astype(np.float32)
        self.detail_map_np = np.array(generate_detail_map(self.original_image)).astype(np.float32)
        print("Base maps generated.")
        self.display_on_canvas(self.canvas_original, self.original_image)
        self.update_display()
        
    def update_display(self):
        if self.original_image is None: return
        
        blend_factor = self.blend_var.get()
        global_strength = self.global_strength_var.get()
        
        char_weight = (1.0 - blend_factor) * global_strength
        detail_weight = blend_factor * global_strength
        
        raw_combined = (char_weight * self.char_map_np) + (detail_weight * self.detail_map_np)
        
        if self.normalize_var.get():
            theoretical_max = global_strength
            if theoretical_max > 0:
                weight_map_01 = np.clip(raw_combined / theoretical_max, 0, 1) ** 0.8
            else:
                weight_map_01 = np.zeros_like(raw_combined)
        else:
            weight_map_01 = np.clip(raw_combined, 0, 1) ** 0.8
        
        # Display weight map (0-1 range)
        weight_map_display = (weight_map_01 * 255).astype(np.uint8)
        weight_pil = Image.fromarray(weight_map_display)
        
        if self.show_heat_map_var.get():
            # Generate heatmap showing effective modulation
            noise_pil, min_mod, max_mod = generate_modulation_heatmap(weight_map_01, global_strength)
        else:
            noise_pil = generate_blocky_noise(weight_map_01, global_strength, self.block_size_var.get())
            min_mod = 1.0 + (weight_map_01.min() * global_strength)
            max_mod = 1.0 + (weight_map_01.max() * global_strength)
        
        # Update statistics
        self.stats_label.config(text=f"[{weight_map_01.min():.2f}, {weight_map_01.max():.2f}]")
        self.modulation_label.config(text=f"[{min_mod:.2f}, {max_mod:.2f}]")
        
        # Update canvases
        self.display_on_canvas(self.canvas_weight, weight_pil)
        self.display_on_canvas(self.canvas_noise, noise_pil)
        
    def display_on_canvas(self, canvas, pil_image):
        if pil_image is None: return
        canvas_w, canvas_h = canvas.winfo_width(), canvas.winfo_height()
        if canvas_w < 2 or canvas_h < 2:
            self.after(50, lambda: self.display_on_canvas(canvas, pil_image))
            return
        img_w, img_h = pil_image.size
        aspect = img_w / img_h
        if canvas_w / canvas_h > aspect:
            new_h = canvas_h; new_w = int(new_h * aspect)
        else:
            new_w = canvas_w; new_h = int(new_w / aspect)
        img_resized = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)
        canvas.delete("all")
        canvas.create_image(canvas_w/2, canvas_h/2, anchor=tk.CENTER, image=img_tk)
        canvas.image = img_tk

if __name__ == "__main__":
    app = SimpleVisualizer()
    app.mainloop()