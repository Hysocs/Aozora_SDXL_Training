import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
# Regex to parse your training logs
pattern = re.compile(
    r"--- Step:\s*(\d+) ---\s*"
    r"Loss:\s*([\d.]+)\s*\|\s*LR:\s*([\deE\+\-\.]+)\s*"
    r"Grad Norm:\s*([\d.]+)\s*\|\s*Max Grad:\s*([\deE\+\-\.]+)\s*"
    r"Timesteps:\s*Min:\s*(\d+)\s*\|\s*Mean:\s*([\d.]+)\s*\|\s*Max:\s*(\d+)\s*"
    r"VRAM \(GB\):\s*([\d.]+)",
    re.MULTILINE
)

def extract_data(text: str):
    data = []
    for m in pattern.finditer(text):
        step, loss, lr, grad_norm, max_grad, tmin, tmean, tmax, vram = m.groups()
        data.append({
            "Step": int(step),
            "Loss": float(loss),
            "LR": float(lr),
            "GradNorm": float(grad_norm),
            "MaxGrad": float(max_grad),
            "TimestepMin": int(tmin),
            "TimestepMean": float(tmean),
            "TimestepMax": int(tmax),
            "VRAM": float(vram)
        })
    return pd.DataFrame(data)

def run_extraction():
    raw_text = input_box.get("1.0", tk.END)
    df = extract_data(raw_text)
    output_box.delete("1.0", tk.END)
    if df.empty:
        messagebox.showwarning("No Matches", "Could not find any training step entries.")
    else:
        output_box.insert(tk.END, df.to_string(index=False))
        global extracted_df
        extracted_df = df

def load_file():
    filepath = filedialog.askopenfilename(
        title="Open Log File",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    if filepath:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        input_box.delete("1.0", tk.END)
        input_box.insert(tk.END, text)

def save_output():
    if extracted_df is None or extracted_df.empty:
        messagebox.showwarning("Warning", "No data to save!")
        return
    filepath = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if filepath:
        extracted_df.to_csv(filepath, index=False)
        messagebox.showinfo("Saved", f"Data saved to:\n{filepath}")

def plot_loss():
    if extracted_df is None or extracted_df.empty:
        messagebox.showwarning("Warning", "No data to plot!")
        return
    
    x = extracted_df["Step"]
    y = extracted_df["Loss"]
    
    # Calculate Gaussian smoothing (much better for spiky data)
    if len(y) > 10:
        # Adjust sigma to control smoothing level (higher = smoother)
        sigma = min(3, max(1, len(y) / 20))  # Adaptive smoothing based on data length
        y_smooth = gaussian_filter1d(y, sigma=sigma)
    else:
        y_smooth = y  # Not enough data to smooth
    
    plt.figure(figsize=(10,6))
    plt.plot(x, y, marker="o", linestyle="-", label="Raw Loss", alpha=0.5, markersize=3)
    
    if len(y) > 10:
        plt.plot(x, y_smooth, "g-", linewidth=3, label=f"Smoothed Trend (σ={sigma:.1f})")
    
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Steps with Smoothed Trend")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# GUI setup
root = tk.Tk()
root.title("Training Log Extractor")
root.geometry("1200x800")

extracted_df = None

# Input
tk.Label(root, text="Paste Raw Logs or Load File:").pack(anchor="w", padx=5, pady=2)
input_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=15)
input_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Buttons
btn_frame = tk.Frame(root)
btn_frame.pack(fill=tk.X, padx=5, pady=5)

tk.Button(btn_frame, text="Load File", command=load_file).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="Extract", command=run_extraction).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="Save CSV", command=save_output).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="Plot Loss", command=plot_loss).pack(side=tk.LEFT, padx=5)

# Output
tk.Label(root, text="Extracted Data:").pack(anchor="w", padx=5, pady=2)
output_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=15, bg="#f0f0f0")
output_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

root.mainloop()
