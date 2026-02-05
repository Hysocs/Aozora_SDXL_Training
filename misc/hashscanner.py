import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import hashlib
import requests
import webbrowser
from threading import Thread

class CivitaiHashChecker:
    def __init__(self, root):
        self.root = root
        self.root.title("Civitai Model Hash Checker")

        self.main_frame = tk.Frame(self.root, padx=10, pady=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.select_button = tk.Button(self.main_frame, text="Select Model and Check Hash", command=self.start_check)
        self.select_button.pack(pady=(0, 10))

        self.hash_label = tk.Label(self.main_frame, text="SHA256 Hash:", wraplength=480, justify=tk.LEFT)
        self.hash_label.pack(pady=(0, 5))

        self.results_text = scrolledtext.ScrolledText(self.main_frame, height=15, width=60, wrap=tk.WORD, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def start_check(self):
        filepath = filedialog.askopenfilename(
            title="Select a Model File",
            filetypes=(("Safetensors", "*.safetensors"), ("Checkpoints", "*.ckpt"), ("All files", "*.*"))
        )
        if not filepath:
            return

        self.select_button.config(state=tk.DISABLED)
        self.hash_label.config(text="Calculating hash...")
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Calculating file hash, this may take a moment for large files...\n")
        self.results_text.config(state=tk.DISABLED)
        self.root.update_idletasks()

        thread = Thread(target=self.check_model, args=(filepath,))
        thread.start()

    def check_model(self, filepath):
        try:
            sha256_hash = self.calculate_sha256(filepath)
            self.root.after(0, self.update_hash_label, sha256_hash)

            self.root.after(0, self.update_results, "Searching on Civitai...\n")
            model_info = self.check_civitai_hash(sha256_hash)

            self.root.after(0, self.display_results, model_info, sha256_hash)
        except Exception as e:
            self.root.after(0, self.show_error, f"An error occurred: {e}")
        finally:
            self.root.after(0, self.enable_button)

    def calculate_sha256(self, filepath):
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256.update(byte_block)
        return sha256.hexdigest()

    def check_civitai_hash(self, sha256_hash):
        url = f"https://civitai.com/api/v1/model-versions/by-hash/{sha256_hash}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except requests.exceptions.RequestException as e:
            self.show_error(f"An error occurred while contacting the Civitai API: {e}")
            return None

    def update_hash_label(self, sha256_hash):
        self.hash_label.config(text=f"SHA256 Hash: {sha256_hash}")

    def update_results(self, message):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, message)
        self.results_text.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def display_results(self, model_info, sha256_hash):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        if model_info:
            model_id = model_info.get("modelId")
            model_version_id = model_info.get("id")
            model_url = f"https://civitai.com/models/{model_id}"
            model_version_url = f"{model_url}?modelVersionId={model_version_id}"

            self.results_text.insert(tk.END, "Model Found!\n\n")
            self.results_text.insert(tk.END, f"Model Name: {model_info.get('model', {}).get('name', 'N/A')}\n")
            self.results_text.insert(tk.END, f"Version: {model_info.get('name', 'N/A')}\n\n")

            self.add_link("Model Page: ", model_url)
            self.add_link("Version Page: ", model_version_url)
        else:
            self.results_text.insert(tk.END, f"No matching models found on Civitai for the hash: {sha256_hash}")
        self.results_text.config(state=tk.DISABLED)

    def add_link(self, label_text, url):
        self.results_text.insert(tk.END, label_text)
        link_start = self.results_text.index(tk.END + f"-{len(url)}c")
        self.results_text.insert(tk.END, url)
        link_end = self.results_text.index(tk.END)
        self.results_text.insert(tk.END, "\n")
        self.results_text.tag_add(url, link_start, link_end)
        self.results_text.tag_config(url, foreground="blue", underline=True)
        self.results_text.tag_bind(url, "<Button-1>", lambda e, u=url: webbrowser.open_new(u))

    def show_error(self, message):
        messagebox.showerror("Error", message)

    def enable_button(self):
        self.select_button.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = CivitaiHashChecker(root)
    root.mainloop()