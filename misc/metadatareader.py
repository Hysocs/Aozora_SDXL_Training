import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from safetensors.torch import load_file, save_file
from safetensors import safe_open
import json
import os

class MetadataEditorApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Safetensors Metadata Editor")
        self.geometry("800x600")
        
        # Configure grid weights
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.file_path = None
        self.metadata = {}

        # Top frame
        self.top_frame = tk.Frame(self, bd=1, relief="solid")
        self.top_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.top_frame.grid_columnconfigure(1, weight=1)

        self.load_button = tk.Button(
            self.top_frame, 
            text="Load Model", 
            command=self.load_model,
            width=12,
            relief="raised",
            bd=2
        )
        self.load_button.grid(row=0, column=0, padx=(5, 5), pady=5)

        self.file_label = tk.Label(
            self.top_frame, 
            text="No model loaded", 
            anchor="w",
            padx=5
        )
        self.file_label.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Text widget with scrollbar
        self.text_frame = tk.Frame(self)
        self.text_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.text_frame.grid_rowconfigure(0, weight=1)
        self.text_frame.grid_columnconfigure(0, weight=1)

        self.textbox = tk.Text(
            self.text_frame, 
            wrap="word", 
            font=("Courier New", 13),
            undo=True,
            relief="sunken",
            bd=2
        )
        self.textbox.grid(row=0, column=0, sticky="nsew")

        # Add scrollbar
        self.scrollbar = tk.Scrollbar(self.text_frame, command=self.textbox.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.textbox.config(yscrollcommand=self.scrollbar.set)

        self.textbox.insert("1.0", "Load a .safetensors file to see its metadata.")
        self.textbox.config(state="disabled")

        # Bottom frame
        self.bottom_frame = tk.Frame(self, bd=1, relief="solid")
        self.bottom_frame.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")
        self.bottom_frame.grid_columnconfigure(0, weight=1)

        self.save_button = tk.Button(
            self.bottom_frame, 
            text="Save Metadata to New File", 
            command=self.save_metadata, 
            state="disabled",
            relief="raised",
            bd=2
        )
        self.save_button.grid(row=0, column=0, padx=5, pady=5)


    def load_model(self):
        self.file_path = filedialog.askopenfilename(
            title="Select a Safetensors model file",
            filetypes=[("Safetensors files", "*.safetensors")]
        )
        if not self.file_path:
            return

        try:
            with safe_open(self.file_path, framework="pt", device="cpu") as f:
                self.metadata = f.metadata()
            if self.metadata is None:
                self.metadata = {}

            self.file_label.config(text=os.path.basename(self.file_path))
            self.textbox.config(state="normal")
            self.textbox.delete("1.0", "end")
            pretty_metadata = json.dumps(self.metadata, indent=4, sort_keys=True)
            self.textbox.insert("1.0", pretty_metadata)
            self.save_button.config(state="normal")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load metadata from file:\n{e}")
            self.file_label.config(text="Failed to load model.")
            self.save_button.config(state="disabled")

    def save_metadata(self):
        """
        MODIFIED: Saves the tensors and new metadata to a NEW file to avoid OS-level file locking issues.
        """
        if not self.file_path:
            messagebox.showwarning("Warning", "No file is loaded.")
            return

        try:
            edited_metadata_str = self.textbox.get("1.0", "end")
            new_metadata = json.loads(edited_metadata_str)

            # --- KEY CHANGE IS HERE ---
            # 1. Define a new output path.
            # Example: "mymodel.safetensors" becomes "mymodel_updated.safetensors"
            base, ext = os.path.splitext(self.file_path)
            new_file_path = f"{base}_updated{ext}"

            # 2. Load tensors from the original file. This is what creates the OS lock.
            tensors = load_file(self.file_path, device="cpu")

            # 3. Save the tensors and new metadata to the NEW file path.
            save_file(tensors, new_file_path, metadata=new_metadata)

            messagebox.showinfo("Success", f"Metadata saved successfully to a new file!\n\nOriginal: {os.path.basename(self.file_path)}\nNew File: {os.path.basename(new_file_path)}")

        except json.JSONDecodeError:
            messagebox.showerror("Invalid JSON", "The metadata is not in a valid JSON format. Please correct the syntax before saving.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while saving:\n{e}")

if __name__ == "__main__":
    app = MetadataEditorApp()
    app.mainloop()