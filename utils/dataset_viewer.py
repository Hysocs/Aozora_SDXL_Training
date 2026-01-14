import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path
import pyperclip
from transformers import CLIPTokenizer

class DatasetViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Token Viewer")
        self.root.geometry("1200x800")
        
        # Initialize tokenizer (SDXL uses CLIP ViT-L tokenizer)
        print("Loading tokenizer...")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        
        self.image_paths = []
        self.current_page = 0
        self.images_per_page = 12
        self.thumbnail_size = (256, 256)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Top controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Button(
            control_frame, 
            text="Select Dataset Folder", 
            command=self.load_dataset
        ).pack(side=tk.LEFT, padx=5)
        
        self.info_label = ttk.Label(control_frame, text="No dataset loaded")
        self.info_label.pack(side=tk.LEFT, padx=20)
        
        # Navigation
        nav_frame = ttk.Frame(self.root, padding="10")
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.prev_btn = ttk.Button(
            nav_frame, 
            text="◀ Previous", 
            command=self.prev_page,
            state=tk.DISABLED
        )
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.page_label = ttk.Label(nav_frame, text="Page 0/0")
        self.page_label.pack(side=tk.LEFT, padx=20)
        
        self.next_btn = ttk.Button(
            nav_frame, 
            text="Next ▶", 
            command=self.next_page,
            state=tk.DISABLED
        )
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        # Image grid with scrollbar
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(canvas_frame, bg="gray20")
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.grid_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.grid_frame, anchor=tk.NW)
        
        # Bind resize
        self.grid_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        
        # Mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        
    def on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def on_canvas_resize(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)
        
    def load_dataset(self):
        folder = filedialog.askdirectory(title="Select Dataset Folder")
        if not folder:
            return
            
        folder_path = Path(folder)
        
        # Find all images
        extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        self.image_paths = []
        
        for ext in extensions:
            self.image_paths.extend(folder_path.rglob(f"*{ext}"))
        
        if not self.image_paths:
            messagebox.showwarning("No Images", "No images found in selected folder")
            return
            
        self.image_paths.sort()
        self.current_page = 0
        
        self.info_label.config(text=f"Loaded {len(self.image_paths)} images from {folder_path.name}")
        self.update_display()
        
    def get_caption_for_image(self, image_path):
        """Load caption from .txt file or use filename"""
        txt_path = image_path.with_suffix('.txt')
        
        if txt_path.exists():
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                print(f"Error reading {txt_path}: {e}")
                
        # Fallback to filename
        return image_path.stem.replace('_', ' ')
        
    def tokenize_caption(self, caption):
        """Tokenize and return first 77 tokens as text"""
        # Tokenize
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        # Get token IDs
        token_ids = tokens.input_ids[0].tolist()
        
        # Decode back to text (shows what actually gets used)
        decoded_tokens = [self.tokenizer.decode([token_id], skip_special_tokens=True) for token_id in token_ids]
        
        # Join tokens without adding extra spaces
        cleaned_decoded = "".join(decoded_tokens).replace("</w>", " ").strip()

        return cleaned_decoded, token_ids, len([t for t in token_ids if t != self.tokenizer.pad_token_id])
        
    def on_image_click(self, image_path):
        """Handle image click - copy tokens to clipboard"""
        caption = self.get_caption_for_image(image_path)
        decoded, token_ids, actual_length = self.tokenize_caption(caption)
        
        # Copy decoded text to clipboard
        try:
            pyperclip.copy(decoded)
            
            # Show info dialog
            info_text = f"""Image: {image_path.name}
            
Original Caption:
{caption}

Tokenized (First 77 tokens):
{decoded}

Stats:
- Actual tokens used: {actual_length}/77
- Padding tokens: {77 - actual_length}
- Total tokens: 77 (fixed)

✓ Copied to clipboard!"""
            
            messagebox.showinfo("Token Info", info_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy to clipboard: {e}")
            
    def update_display(self):
        """Display current page of images"""
        # Clear existing widgets
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
            
        # Calculate page bounds
        start_idx = self.current_page * self.images_per_page
        end_idx = min(start_idx + self.images_per_page, len(self.image_paths))
        
        page_images = self.image_paths[start_idx:end_idx]
        
        # Display in grid (4 columns)
        cols = 4
        for idx, img_path in enumerate(page_images):
            row = idx // cols
            col = idx % cols
            
            # Create frame for each image
            img_frame = ttk.Frame(self.grid_frame, relief=tk.RAISED, borderwidth=2)
            img_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            try:
                # Load and create thumbnail
                img = Image.open(img_path)
                img.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Image label (clickable)
                img_label = ttk.Label(img_frame, image=photo, cursor="hand2")
                img_label.image = photo  # Keep reference
                img_label.pack(padx=5, pady=5)
                
                # Bind click event
                img_label.bind("<Button-1>", lambda e, path=img_path: self.on_image_click(path))
                
                # Filename label
                name_label = ttk.Label(
                    img_frame, 
                    text=img_path.name, 
                    wraplength=200,
                    justify=tk.CENTER
                )
                name_label.pack(padx=5, pady=5)
                
                # Caption preview (first 50 chars)
                caption = self.get_caption_for_image(img_path)
                preview = caption[:50] + "..." if len(caption) > 50 else caption
                caption_label = ttk.Label(
                    img_frame,
                    text=preview,
                    wraplength=200,
                    justify=tk.LEFT,
                    foreground="gray"
                )
                caption_label.pack(padx=5, pady=5)
                
            except Exception as e:
                error_label = ttk.Label(img_frame, text=f"Error loading\n{img_path.name}")
                error_label.pack(padx=5, pady=5)
                print(f"Error loading {img_path}: {e}")
        
        # Configure grid weights
        for i in range(cols):
            self.grid_frame.columnconfigure(i, weight=1)
            
        # Update navigation
        total_pages = (len(self.image_paths) + self.images_per_page - 1) // self.images_per_page
        self.page_label.config(text=f"Page {self.current_page + 1}/{total_pages}")
        
        self.prev_btn.config(state=tk.NORMAL if self.current_page > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_page < total_pages - 1 else tk.DISABLED)
        
        # Reset scroll position
        self.canvas.yview_moveto(0)
        
    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.update_display()
            
    def next_page(self):
        total_pages = (len(self.image_paths) + self.images_per_page - 1) // self.images_per_page
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self.update_display()

def main():
    root = tk.Tk()
    app = DatasetViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()