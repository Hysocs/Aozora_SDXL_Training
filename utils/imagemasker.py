import os
import sys
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageTk
from ultralytics import YOLO
from tqdm import tqdm

# --- GUI Components (requires tkinter) ---
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:
    print("Tkinter is not installed. Preview mode is unavailable.")
    tk = None

def create_mask(image, face_model, hand_model, confidence_threshold, image_size, box_padding, mask_color=(255, 0, 0)):
    """
    Generates a mask for a given PIL Image object without saving it.

    Returns:
        PIL.Image: The generated mask image.
    """
    img_width, img_height = image.size
    mask = Image.new("RGB", image.size, (0, 0, 0))
    draw = ImageDraw.Draw(mask)

    # --- Detect Faces with adjusted settings ---
    face_results = face_model(image, conf=confidence_threshold, imgsz=image_size, verbose=False)
    for result in face_results:
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [
                max(0, xyxy[0] - box_padding),
                max(0, xyxy[1] - box_padding),
                min(img_width, xyxy[2] + box_padding),
                min(img_height, xyxy[3] + box_padding)
            ]
            draw.rectangle([x1, y1, x2, y2], fill=mask_color)

    # --- Detect Hands with adjusted settings ---
    hand_results = hand_model(image, conf=confidence_threshold, imgsz=image_size, verbose=False)
    for result in hand_results:
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [
                max(0, xyxy[0] - box_padding),
                max(0, xyxy[1] - box_padding),
                min(img_width, xyxy[2] + box_padding),
                min(img_height, xyxy[3] + box_padding)
            ]
            draw.rectangle([x1, y1, x2, y2], fill=mask_color)
            
    return mask

def show_preview_gui(image_files, models, settings):
    """
    Creates and displays a 2x2 GUI with overlayed mask previews.
    """
    if tk is None:
        print("Cannot show preview because Tkinter is not available.")
        return False

    root = tk.Tk()
    root.title("Mask Preview - Are these settings correct?")
    root.proceed = False 

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    num_images = min(len(image_files), 4)
    if num_images == 0:
        messagebox.showerror("Error", "No images found in the dataset folder to preview.")
        return False
        
    preview_images = random.sample(image_files, num_images)
    
    for i, img_path in enumerate(preview_images):
        row, col = divmod(i, 2)
        
        try:
            original_img = Image.open(img_path).convert("RGB")
            
            # --- FIX IS HERE ---
            # Call create_mask with explicit, lowercase keyword arguments
            mask = create_mask(
                original_img, 
                models['face'], 
                models['hand'],
                confidence_threshold=settings['CONFIDENCE_THRESHOLD'],
                image_size=settings['IMAGE_SIZE'],
                box_padding=settings['BOX_PADDING']
            )
            # --- END FIX ---
            
            blended_img = Image.blend(original_img, mask, alpha=0.4)
            blended_img.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(blended_img)
            
            label = ttk.Label(main_frame, image=photo)
            label.image = photo
            label.grid(row=row, column=col, padx=5, pady=5)

        except Exception as e:
            print(f"Error creating preview for {img_path}: {e}")
            error_label = ttk.Label(main_frame, text=f"Error loading\n{img_path.name}", borderwidth=2, relief="solid")
            error_label.grid(row=row, column=col, padx=5, pady=5, ipadx=50, ipady=100)

    button_frame = ttk.Frame(root, padding="10")
    button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
    root.grid_columnconfigure(0, weight=1)

    def on_proceed():
        root.proceed = True
        root.destroy()

    def on_cancel():
        root.proceed = False
        root.destroy()

    proceed_button = ttk.Button(button_frame, text="Looks Good, Run Full Process", command=on_proceed)
    proceed_button.pack(side=tk.RIGHT, padx=5)
    cancel_button = ttk.Button(button_frame, text="Cancel", command=on_cancel)
    cancel_button.pack(side=tk.RIGHT)
    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.mainloop()
    
    return root.proceed

def run_full_process(image_files, models, settings, output_path):
    """
    Processes all images in the list and saves the masks.
    """
    print(f"Found {len(image_files)} images to process in '{settings['DATASET_DIR']}'.")
    for image_file in tqdm(image_files, desc="Creating masks"):
        try:
            with Image.open(image_file).convert("RGB") as img:
                # --- FIX IS HERE ---
                # Call create_mask with explicit, lowercase keyword arguments
                mask = create_mask(
                    img, 
                    models['face'], 
                    models['hand'],
                    confidence_threshold=settings['CONFIDENCE_THRESHOLD'],
                    image_size=settings['IMAGE_SIZE'],
                    box_padding=settings['BOX_PADDING']
                )
                # --- END FIX ---
                mask_filename = f"{image_file.stem}_mask.png"
                mask.save(output_path / mask_filename)
        except Exception as e:
            print(f"Failed to process {image_file}: {e}")

    print(f"\nMask generation complete. Masks are saved in: {output_path}")

def main():
    # =================================================================================
    # --- CONFIGURATION SETTINGS ---
    # =================================================================================
    settings = {
        "DATASET_DIR": r"C:\Users\Administrator\Pictures\Datasets\by dbmaster",
        "FACE_MODEL_PATH": r"C:\Users\Administrator\Desktop\Aozora_SDXL_Training\utils\Yolo\face_yolov9c.pt",
        "HAND_MODEL_PATH": r"C:\Users\Administrator\Desktop\Aozora_SDXL_Training\utils\Yolo\hand_yolov9c.pt",
        "CONFIDENCE_THRESHOLD": 0.60,
        "IMAGE_SIZE": 1024,
        "BOX_PADDING": 20,
        "RUN_MODE": "preview"
    }
    # =================================================================================

    dataset_path = Path(settings['DATASET_DIR'])
    output_path = dataset_path / "masks"
    output_path.mkdir(exist_ok=True)
    
    for key, path in [('Dataset', settings['DATASET_DIR']), 
                      ('Face Model', settings['FACE_MODEL_PATH']), 
                      ('Hand Model', settings['HAND_MODEL_PATH'])]:
        if not Path(path).exists():
            print(f"ERROR: {key} path not found at '{path}'")
            return

    print("Loading models...")
    models = {
        'face': YOLO(settings['FACE_MODEL_PATH']),
        'hand': YOLO(settings['HAND_MODEL_PATH'])
    }
    print("Models loaded successfully.")

    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    image_files = [p for ext in image_extensions for p in dataset_path.rglob(f"*{ext}") if "_mask" not in p.stem]

    # No longer need to filter settings, we will pass them explicitly
    if settings['RUN_MODE'] == 'preview':
        # Pass all settings to the GUI function
        should_proceed = show_preview_gui(image_files, models, settings)
        if should_proceed:
            print("\nPreview approved. Starting full processing...")
            run_full_process(image_files, models, settings, output_path)
        else:
            print("\nOperation cancelled by user.")
    elif settings['RUN_MODE'] == 'full':
        run_full_process(image_files, models, settings, output_path)
    else:
        print(f"Error: Invalid RUN_MODE '{settings['RUN_MODE']}'. Please choose 'preview' or 'full'.")

if __name__ == "__main__":
    main()