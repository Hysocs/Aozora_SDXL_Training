import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

class AlphaFixVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Training Data Preprocessing Check")
        self.root.geometry("1300x700")
        self.root.configure(bg="#f0f0f0")

        # --- Top Control Bar ---
        control_frame = tk.Frame(root, bg="#e0e0e0", pady=10)
        control_frame.pack(fill="x")

        btn_load = tk.Button(
            control_frame, 
            text="📂 Load Image (PNG/WEBP)", 
            command=self.load_image,
            font=("Segoe UI", 11, "bold"),
            bg="#007acc", fg="white", padx=15, pady=5
        )
        btn_load.pack()
        
        info_lbl = tk.Label(
            control_frame,
            text="Load a transparent image to see how the model interprets the Alpha Channel.",
            bg="#e0e0e0", font=("Segoe UI", 9)
        )
        info_lbl.pack(pady=5)

        # --- Main Display Area ---
        self.canvas_frame = tk.Frame(root, bg="#f0f0f0")
        self.canvas_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Grid configuration
        self.canvas_frame.columnconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(1, weight=1)
        self.canvas_frame.rowconfigure(1, weight=1)

        # --- LEFT SIDE: THE PROBLEM ---
        lbl_bad_title = tk.Label(
            self.canvas_frame, 
            text="❌ BEFORE (Current Code)\nimg.convert('RGB')", 
            font=("Segoe UI", 12, "bold"), fg="#cc0000", bg="#f0f0f0"
        )
        lbl_bad_title.grid(row=0, column=0, pady=(0, 10))

        self.lbl_bad_desc = tk.Label(
            self.canvas_frame,
            text="Transparency becomes BLACK.\nNote the jagged/dark 'halo' around edges.",
            font=("Segoe UI", 9), bg="#f0f0f0", justify="center"
        )
        self.lbl_bad_desc.grid(row=2, column=0, pady=10)

        self.panel_bad = tk.Label(self.canvas_frame, bg="#333333", text="No Image", fg="white")
        self.panel_bad.grid(row=1, column=0, sticky="nsew", padx=10)

        # --- RIGHT SIDE: THE FIX ---
        lbl_good_title = tk.Label(
            self.canvas_frame, 
            text="✅ AFTER (Proposed Fix)\nComposited on White", 
            font=("Segoe UI", 12, "bold"), fg="#008800", bg="#f0f0f0"
        )
        lbl_good_title.grid(row=0, column=1, pady=(0, 10))

        self.lbl_good_desc = tk.Label(
            self.canvas_frame,
            text="Transparency becomes WHITE.\nEdges are smooth and blended correctly.",
            font=("Segoe UI", 9), bg="#f0f0f0", justify="center"
        )
        self.lbl_good_desc.grid(row=2, column=1, pady=10)

        self.panel_good = tk.Label(self.canvas_frame, bg="#ffffff", text="No Image", fg="black")
        self.panel_good.grid(row=1, column=1, sticky="nsew", padx=10)

    def load_image_with_alpha_handling(self, img, background_color=(255, 255, 255)):
        """The fix logic extracted from the proposed solution."""
        if img.mode == 'P' and 'transparency' in img.info:
            img = img.convert('RGBA')
        
        if img.mode in ('RGBA', 'LA'):
            bg = Image.new('RGB', img.size, background_color)
            bg.paste(img, mask=img.split()[-1])
            return bg
        else:
            return img.convert('RGB')

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.webp;*.jpg;*.jpeg;*.bmp;*.tiff")]
        )
        if not file_path:
            return

        try:
            # Open original
            original_pil = Image.open(file_path)
            
            # --- 1. Simulate the "Before" (The Issue) ---
            # This simulates exactly what `img.convert("RGB")` does in your current script
            bad_pil = original_pil.copy().convert("RGB")

            # --- 2. Apply the "After" (The Fix) ---
            good_pil = self.load_image_with_alpha_handling(original_pil.copy())

            # --- Resize for GUI display (Keep Aspect Ratio) ---
            # We calculate size based on the panel size
            display_h = 500
            aspect = original_pil.width / original_pil.height
            display_w = int(display_h * aspect)

            # High quality resize for preview
            bad_preview = bad_pil.resize((display_w, display_h), Image.Resampling.LANCZOS)
            good_preview = good_pil.resize((display_w, display_h), Image.Resampling.LANCZOS)

            # Convert to ImageTk
            self.tk_bad = ImageTk.PhotoImage(bad_preview)
            self.tk_good = ImageTk.PhotoImage(good_preview)

            # Update Labels
            self.panel_bad.config(image=self.tk_bad, text="")
            self.panel_good.config(image=self.tk_good, text="")
            
        except Exception as e:
            tk.messagebox.showerror("Error", f"Could not load image:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AlphaFixVisualizer(root)
    root.mainloop()