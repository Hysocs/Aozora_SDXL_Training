import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os

class RobustImageComparer:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Epoch Comparer - Robust Edition")
        self.root.geometry("1400x900")
        
        # --- Data Storage ---
        self.source_images = {}      # path -> original PIL Image
        self.epoch_paths = []        # List of loaded paths
        self.selected_paths = []     # Currently comparing [left, right]
        
        # --- View State ---
        self.zoom = 1.0
        self.split = 0.5             # 0.0 to 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.view_size = (800, 600)  # Current display resolution for images
        
        # --- Buffers (resized to screen) ---
        self.buf_left = None         # PIL Image (RGB, screen size)
        self.buf_right = None        # PIL Image
        self.photo_left = None       # PhotoImage (must keep ref!)
        self.photo_right = None      # PhotoImage
        
        # --- Canvas Items ---
        self.cvs_left = None         # Canvas image id
        self.cvs_right = None        # Canvas image id
        self.cvs_line = None         # Slider line id
        
        self.setup_ui()
        self.bind_events()
        
        # Delayed setup to ensure canvas has size
        self.root.after(100, self.on_resize)
        
    def setup_ui(self):
        # Toolbar
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(toolbar, text="Load Epochs (Ctrl+O)", command=self.load_images).pack(side='left', padx=2)
        ttk.Separator(toolbar, orient='vertical').pack(side='left', fill='y', padx=5)
        
        self.lbl_zoom = ttk.Label(toolbar, text="Zoom: 100%")
        self.lbl_zoom.pack(side='left', padx=5)
        
        ttk.Button(toolbar, text="Fit", command=self.zoom_fit).pack(side='left', padx=2)
        ttk.Button(toolbar, text="100%", command=lambda: self.set_zoom(1.0)).pack(side='left', padx=2)
        ttk.Button(toolbar, text="Reset All", command=self.reset_all).pack(side='left', padx=2)
        
        self.lbl_info = ttk.Label(toolbar, text="No images loaded", font=('Segoe UI', 9, 'bold'))
        self.lbl_info.pack(side='right', padx=10)
        
        # Selection Tabs
        tab_frame = ttk.LabelFrame(self.root, text="Select 2 Epochs to Compare")
        tab_frame.pack(fill='x', padx=5, pady=2)
        
        self.tabs_canvas = tk.Canvas(tab_frame, height=50, bg='#2b2b2b', highlightthickness=0)
        self.tabs_scroll = ttk.Scrollbar(tab_frame, orient="horizontal", command=self.tabs_canvas.xview)
        self.tabs_canvas.configure(xscrollcommand=self.tabs_scroll.set)
        
        self.tabs_scroll.pack(side='bottom', fill='x')
        self.tabs_canvas.pack(side='top', fill='x', expand=True)
        
        self.tab_container = ttk.Frame(self.tabs_canvas)
        self.tab_canvas_window = self.tabs_canvas.create_window((0, 0), window=self.tab_container, anchor='nw')
        
        # Main Canvas
        self.canvas = tk.Canvas(
            self.root, 
            bg='#1a1a1a',
            highlightthickness=2,
            highlightbackground='#444'
        )
        self.canvas.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Status bar
        self.status = ttk.Label(self.root, text="Ready", relief='sunken', anchor='w')
        self.status.pack(fill='x', side='bottom')
        
    def bind_events(self):
        self.root.bind("<Control-o>", lambda e: self.load_images())
        self.root.bind("<Left>", lambda e: self.nudge_split(-0.01))
        self.root.bind("<Right>", lambda e: self.nudge_split(0.01))
        self.root.bind("<plus>", lambda e: self.zoom_in())
        self.root.bind("<minus>", lambda e: self.zoom_out())
        self.root.bind("<r>", lambda e: self.reset_all())
        
        # Mouse
        self.canvas.bind("<Button-1>", self.on_left_down)
        self.canvas.bind("<B1-Motion>", self.on_left_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_up)
        
        self.canvas.bind("<Button-3>", self.on_right_down)
        self.canvas.bind("<B3-Motion>", self.on_right_move)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_up)
        
        # Wheel
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", self.on_wheel)
        self.canvas.bind("<Button-5>", self.on_wheel)
        
        # Resize
        self.canvas.bind("<Configure>", self.on_resize)
        
    def load_images(self):
        paths = filedialog.askopenfilenames(
            title="Select Epoch Images",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All Files", "*.*")]
        )
        if not paths:
            return
            
        # Clear previous
        self.clear_view()
        self.epoch_paths = list(paths)
        self.source_images = {}
        
        # Load originals (heavy lifting once)
        errors = []
        for p in self.epoch_paths:
            try:
                img = Image.open(p)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                self.source_images[p] = img
            except Exception as e:
                errors.append(f"{os.path.basename(p)}: {e}")
                
        if errors:
            messagebox.showerror("Load Errors", "\n".join(errors))
            
        self.build_tabs()
        
        # Auto-select first two
        if len(self.epoch_paths) >= 2:
            self.select_path(self.epoch_paths[0])
            self.select_path(self.epoch_paths[1])
        elif len(self.epoch_paths) == 1:
            self.select_path(self.epoch_paths[0])
            
    def build_tabs(self):
        # Clear
        for w in self.tab_container.winfo_children():
            w.destroy()
            
        self.tab_buttons = {}
        
        for path in self.epoch_paths:
            name = os.path.basename(path)
            short = name[:15] + "..." if len(name) > 15 else name
            
            btn = tk.Label(
                self.tab_container,
                text=short,
                bg='#404040',
                fg='white',
                padx=12,
                pady=6,
                cursor='hand2',
                relief='raised',
                font=('Segoe UI', 9)
            )
            btn.pack(side='left', padx=2, pady=2)
            btn.bind("<Button-1>", lambda e, p=path: self.select_path(p))
            self.tab_buttons[path] = btn
            
        self.update_tab_scroll()
        
    def update_tab_scroll(self):
        self.tab_container.update_idletasks()
        self.tabs_canvas.configure(scrollregion=self.tabs_canvas.bbox("all"))
        
    def select_path(self, path):
        if path in self.selected_paths:
            self.selected_paths.remove(path)
            self.tab_buttons[path].config(bg='#404040', relief='raised')
        else:
            if len(self.selected_paths) >= 2:
                old = self.selected_paths.pop(0)
                self.tab_buttons[old].config(bg='#404040', relief='raised')
            self.selected_paths.append(path)
            self.tab_buttons[path].config(bg='#2ecc71', fg='white', relief='sunken')
            
        self.prepare_comparison()
        
    def prepare_comparison(self):
        """Generate screen-sized buffers for current zoom level"""
        self.clear_canvas_items()
        
        if len(self.selected_paths) == 0:
            self.show_placeholder("Select 2 epoch images from the tabs above")
            return
            
        if len(self.selected_paths) == 1:
            self.show_single(self.selected_paths[0])
            return
            
        # Dual mode
        left_path, right_path = self.selected_paths
        
        # Get canvas dims
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        
        # Calculate display size to fit both
        img1 = self.source_images[left_path]
        img2 = self.source_images[right_path]
        
        w1, h1 = img1.size
        w2, h2 = img2.size
        
        # Scale to fit canvas with current zoom
        scale = min(
            (cw * self.zoom) / max(w1, w2),
            (ch * self.zoom) / max(h1, h2)
        )
        
        new_w = int(max(w1, w2) * scale)
        new_h = int(max(h1, h2) * scale)
        self.view_size = (new_w, new_h)
        
        # Resize to buffers (high quality)
        r1 = img1.resize((int(w1*scale), int(h1*scale)), Image.Resampling.LANCZOS)
        r2 = img2.resize((int(w2*scale), int(h2*scale)), Image.Resampling.LANCZOS)
        
        # Pad to same size (centered) so they align perfectly
        self.buf_left = Image.new('RGB', (new_w, new_h), (30,30,30))
        self.buf_right = Image.new('RGB', (new_w, new_h), (30,30,30))
        
        off_x1 = (new_w - r1.width) // 2
        off_y1 = (new_h - r1.height) // 2
        off_x2 = (new_w - r2.width) // 2
        off_y2 = (new_h - r2.height) // 2
        
        self.buf_left.paste(r1, (off_x1, off_y1))
        self.buf_right.paste(r2, (off_x2, off_y2))
        
        # Create initial photos (full image)
        self.update_photos_full()
        
        # Create canvas items
        cx = cw // 2 + self.pan_x
        cy = ch // 2 + self.pan_y
        
        self.cvs_left = self.canvas.create_image(cx, cy, image=self.photo_left, anchor='center')
        self.cvs_right = self.canvas.create_image(cx, cy, image=self.photo_right, anchor='center')
        
        # Slider line (on top)
        self.cvs_line = self.canvas.create_line(0, 0, 0, 0, fill='#FFD700', width=3)
        
        self.update_split_view()
        self.update_info(left_path, right_path)
        
    def update_photos_full(self):
        """Create PhotoImages from current buffers"""
        self.photo_left = ImageTk.PhotoImage(self.buf_left)
        self.photo_right = ImageTk.PhotoImage(self.buf_right)
        
    def update_split_view(self):
        """Crop buffers at split point and update canvas images"""
        if not self.buf_left or not self.buf_right:
            return
            
        w, h = self.view_size
        split_x = int(w * self.split)
        
        # Prevent zero-width crops
        split_x = max(1, min(w-1, split_x))
        
        # Crop
        left_crop = self.buf_left.crop((0, 0, split_x, h))
        right_crop = self.buf_right.crop((split_x, 0, w, h))
        
        # Convert to PhotoImage (must keep refs!)
        self.photo_left = ImageTk.PhotoImage(left_crop)
        self.photo_right = ImageTk.PhotoImage(right_crop)
        
        # Update canvas
        self.canvas.itemconfig(self.cvs_left, image=self.photo_left)
        self.canvas.itemconfig(self.cvs_right, image=self.photo_right)
        
        # Update positions to align seamlessly
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        cx = cw // 2 + self.pan_x
        cy = ch // 2 + self.pan_y
        
        # Left image: centered on its visible portion
        left_cx = cx - w//2 + split_x//2
        self.canvas.coords(self.cvs_left, left_cx, cy)
        
        # Right image: centered on its visible portion  
        right_cx = cx - w//2 + split_x + (w - split_x)//2
        self.canvas.coords(self.cvs_right, right_cx, cy)
        
        # Update slider line position
        line_x = cx - w//2 + split_x
        self.canvas.coords(self.cvs_line, line_x, cy - h//2, line_x, cy + h//2)
        self.slider_x = line_x  # Store for hit testing
        
    def show_single(self, path):
        img = self.source_images[path]
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        
        w, h = img.size
        scale = min((cw * self.zoom) / w, (ch * self.zoom) / h)
        
        resized = img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
        self.photo_left = ImageTk.PhotoImage(resized)
        
        cx = cw // 2 + self.pan_x
        cy = ch // 2 + self.pan_y
        self.canvas.create_image(cx, cy, image=self.photo_left)
        self.lbl_info.config(text=f"Viewing: {os.path.basename(path)}")
        
    def show_placeholder(self, text):
        cx = self.canvas.winfo_width() // 2
        cy = self.canvas.winfo_height() // 2
        self.canvas.create_text(cx, cy, text=text, fill='#888', font=('Segoe UI', 14))
        
    def clear_canvas_items(self):
        self.canvas.delete('all')
        self.cvs_left = self.cvs_right = self.cvs_line = None
        self.buf_left = self.buf_right = None
        self.photo_left = self.photo_right = None
        
    def clear_view(self):
        self.clear_canvas_items()
        self.selected_paths = []
        self.epoch_paths = []
        self.source_images = {}
        
    # --- Interactions ---
    
    def on_left_down(self, e):
        self.drag_last = (e.x, e.y)
        
        # Check if clicking slider line (within 20px)
        if hasattr(self, 'slider_x') and abs(e.x - self.slider_x) < 20:
            self.drag_mode = 'split'
            self.canvas.config(cursor='sb_h_double_arrow')
        else:
            self.drag_mode = None
            
    def on_left_move(self, e):
        if self.drag_mode == 'split' and self.buf_left:
            dx = e.x - self.drag_last[0]
            w = self.view_size[0]
            
            # Convert pixel movement to split delta
            delta = dx / w
            self.split = max(0.0, min(1.0, self.split + delta))
            self.drag_last = (e.x, e.y)
            
            self.update_split_view()
            
    def on_left_up(self, e):
        self.drag_mode = None
        self.canvas.config(cursor='crosshair')
        
    def on_right_down(self, e):
        self.drag_mode = 'pan'
        self.pan_start = (e.x, e.y)
        self.pan_start_pos = (self.pan_x, self.pan_y)
        self.canvas.config(cursor='fleur')
        
    def on_right_move(self, e):
        if self.drag_mode == 'pan':
            dx = e.x - self.pan_start[0]
            dy = e.y - self.pan_start[1]
            self.pan_x = self.pan_start_pos[0] + dx
            self.pan_y = self.pan_start_pos[1] + dy
            self.update_split_view()
            
    def on_right_up(self, e):
        self.drag_mode = None
        self.canvas.config(cursor='crosshair')
        
    def on_wheel(self, e):
        if hasattr(e, 'delta') and e.delta > 0:
            factor = 1.1
        elif hasattr(e, 'delta') and e.delta < 0:
            factor = 0.9
        elif e.num == 4:
            factor = 1.1
        else:
            factor = 0.9
            
        new_zoom = max(0.1, min(10.0, self.zoom * factor))
        if new_zoom != self.zoom:
            self.zoom = new_zoom
            self.lbl_zoom.config(text=f"Zoom: {int(self.zoom*100)}%")
            self.prepare_comparison()  # Rebuild buffers at new zoom
            
    def zoom_in(self):
        self.zoom = min(10.0, self.zoom * 1.2)
        self.prepare_comparison()
        
    def zoom_out(self):
        self.zoom = max(0.1, self.zoom / 1.2)
        self.prepare_comparison()
        
    def zoom_fit(self):
        self.zoom = 1.0
        self.pan_x = self.pan_y = 0
        self.prepare_comparison()
        
    def set_zoom(self, z):
        self.zoom = z
        self.lbl_zoom.config(text=f"Zoom: {int(z*100)}%")
        self.prepare_comparison()
        
    def reset_all(self):
        self.split = 0.5
        self.zoom_fit()
        
    def nudge_split(self, delta):
        if len(self.selected_paths) == 2:
            self.split = max(0.0, min(1.0, self.split + delta))
            self.update_split_view()
            
    def update_info(self, left, right):
        self.lbl_info.config(text=f"{os.path.basename(left)} ← → {os.path.basename(right)}")
        self.status.config(text=f"Split: {int(self.split*100)}% | Pan: ({self.pan_x}, {self.pan_y}) | Use ← → arrows to nudge")
        
    def on_resize(self, event=None):
        # Debounce
        if hasattr(self, '_resize_after'):
            self.root.after_cancel(self._resize_after)
        self._resize_after = self.root.after(200, self.prepare_comparison)

if __name__ == "__main__":
    root = tk.Tk()
    app = RobustImageComparer(root)
    root.mainloop()