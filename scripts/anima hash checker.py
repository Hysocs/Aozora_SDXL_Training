import hashlib
import json
import os
import queue
import threading
import ssl
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

try:
    import certifi
except Exception:
    certifi = None

APP_TITLE = "Civitai Model Hash Comparer"
API_BASE = "https://civitai.com/api/v1"
CHUNK_SIZE = 1024 * 1024 * 8


def sha256_file(path, progress_cb=None):
    h = hashlib.sha256()
    total = os.path.getsize(path)
    read = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
            if progress_cb:
                progress_cb(read, total)
    return h.hexdigest().upper()


def model_hash_old(path):
    # Legacy A1111-style model hash: SHA256 of bytes 0x100000..0x110000, first 8 hex chars.
    # Not a full identity hash, but useful for older metadata.
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(0x100000)
        h.update(f.read(0x10000))
    return h.hexdigest()[:8].upper()


def add_auth_headers(headers, token):
    token = (token or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def make_ssl_context(allow_insecure_ssl=False):
    if allow_insecure_ssl:
        return ssl._create_unverified_context()

    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())

    return ssl.create_default_context()


def api_get_json(url, token="", timeout=30, allow_insecure_ssl=False):
    req = Request(url, headers=add_auth_headers({"User-Agent": APP_TITLE}, token))
    context = make_ssl_context(allow_insecure_ssl=allow_insecure_ssl)
    with urlopen(req, timeout=timeout, context=context) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def get_version_by_hash(hash_value, token="", allow_insecure_ssl=False):
    url = f"{API_BASE}/model-versions/by-hash/{quote(hash_value)}"
    try:
        return api_get_json(url, token=token, allow_insecure_ssl=allow_insecure_ssl)
    except HTTPError as e:
        if e.code == 404:
            return None
        raise


def same_hash(file_obj, hash_value):
    hashes = file_obj.get("hashes") or {}
    target = hash_value.upper()
    for key, value in hashes.items():
        if str(value).upper() == target:
            return key
    return None


def version_url(version):
    model_id = version.get("modelId")
    version_id = version.get("id")
    if model_id and version_id:
        return f"https://civitai.com/models/{model_id}?modelVersionId={version_id}"
    return ""


class HashComparerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1150x720")
        self.minsize(900, 560)
        self.q = queue.Queue()
        self.worker = None
        self.selected_path = tk.StringVar()
        self.token = tk.StringVar()
        self.allow_insecure_ssl = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Select a model file to begin.")
        self.sha256_value = tk.StringVar(value="")
        self.old_hash_value = tk.StringVar(value="")
        self.match_count = tk.StringVar(value="0")
        self._build_ui()
        self.after(100, self._poll_queue)

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Model file:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.selected_path).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(top, text="Browse", command=self.browse).grid(row=0, column=2)
        ttk.Button(top, text="Hash + Search Civitai", command=self.start_scan).grid(row=0, column=3, padx=(8, 0))
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="API token, optional:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.token, show="*").grid(row=1, column=1, sticky="ew", padx=6, pady=(8, 0))
        ttk.Checkbutton(
            top,
            text="Disable SSL verification, only if certifi still fails",
            variable=self.allow_insecure_ssl,
        ).grid(row=1, column=2, columnspan=2, sticky="w", pady=(8, 0))

        info = ttk.Frame(self, padding=(10, 0, 10, 8))
        info.pack(fill="x")
        ttk.Label(info, text="SHA256:").grid(row=0, column=0, sticky="w")
        ttk.Entry(info, textvariable=self.sha256_value, state="readonly").grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(info, text="Copy SHA256", command=lambda: self.copy_var(self.sha256_value)).grid(row=0, column=2)
        ttk.Label(info, text="Legacy hash:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(info, textvariable=self.old_hash_value, state="readonly", width=20).grid(row=1, column=1, sticky="w", padx=6, pady=(6, 0))
        ttk.Label(info, text="Matching files found:").grid(row=1, column=2, sticky="e", padx=(20, 4), pady=(6, 0))
        ttk.Label(info, textvariable=self.match_count).grid(row=1, column=3, sticky="w", pady=(6, 0))
        info.columnconfigure(1, weight=1)

        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Label(self, textvariable=self.status, padding=(10, 0, 10, 8)).pack(fill="x")

        columns = ("hashType", "model", "version", "file", "type", "size", "modelId", "versionId", "url")
        self.tree = ttk.Treeview(self, columns=columns, show="headings")
        headings = {
            "hashType": "Matched Hash",
            "model": "Model",
            "version": "Version",
            "file": "File",
            "type": "File Type",
            "size": "Size MB",
            "modelId": "Model ID",
            "versionId": "Version ID",
            "url": "URL",
        }
        widths = {
            "hashType": 110,
            "model": 220,
            "version": 170,
            "file": 260,
            "type": 95,
            "size": 80,
            "modelId": 80,
            "versionId": 85,
            "url": 300,
        }
        for col in columns:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=widths[col], anchor="w")
        yscroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        xscroll = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=(0, 10))
        yscroll.pack(side="right", fill="y", pady=(0, 10))
        xscroll.pack(fill="x", padx=10)

        bottom = ttk.Frame(self, padding=(10, 0, 10, 10))
        bottom.pack(fill="x")
        ttk.Button(bottom, text="Copy selected URL", command=self.copy_selected_url).pack(side="left")
        ttk.Button(bottom, text="Export JSON", command=self.export_json).pack(side="left", padx=8)

        self.results = []

    def browse(self):
        path = filedialog.askopenfilename(
            title="Select model file",
            filetypes=[
                ("Model files", "*.safetensors *.ckpt *.pt *.pth *.bin"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.selected_path.set(path)

    def copy_var(self, var):
        value = var.get()
        if value:
            self.clipboard_clear()
            self.clipboard_append(value)
            self.status.set("Copied to clipboard.")

    def copy_selected_url(self):
        item = self.tree.focus()
        if not item:
            return
        values = self.tree.item(item, "values")
        if values:
            self.clipboard_clear()
            self.clipboard_append(values[-1])
            self.status.set("Copied URL to clipboard.")

    def clear_results(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.results = []
        self.match_count.set("0")

    def start_scan(self):
        path = self.selected_path.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror(APP_TITLE, "Select a valid model file first.")
            return
        if self.worker and self.worker.is_alive():
            messagebox.showinfo(APP_TITLE, "A scan is already running.")
            return
        self.clear_results()
        self.progress["value"] = 0
        self.sha256_value.set("")
        self.old_hash_value.set("")
        self.status.set("Hashing file...")
        self.worker = threading.Thread(
            target=self._scan_worker,
            args=(path, self.token.get(), self.allow_insecure_ssl.get()),
            daemon=True,
        )
        self.worker.start()

    def _scan_worker(self, path, token, allow_insecure_ssl=False):
        try:
            def progress(read, total):
                self.q.put(("progress", read, total))

            sha = sha256_file(path, progress_cb=progress)
            old = model_hash_old(path)
            self.q.put(("hashes", sha, old))

            # Query strongest first. Civitai says by-hash supports SHA256 and Auto-style hashes;
            # using both helps catch older/short hash metadata.
            queries = [("SHA256", sha), ("Legacy/AutoV1-ish", old)]
            seen_versions = set()
            rows = []

            for label, hv in queries:
                self.q.put(("status", f"Querying Civitai by {label}..."))
                data = get_version_by_hash(hv, token=token, allow_insecure_ssl=allow_insecure_ssl)
                if not data:
                    continue

                version_id = data.get("id")
                if version_id in seen_versions:
                    # Still check files for this specific hash label, but don't duplicate version rows by default.
                    pass
                seen_versions.add(version_id)

                model_name = data.get("model", {}).get("name") or data.get("modelName") or ""
                model_id = data.get("modelId") or data.get("model", {}).get("id") or ""
                version_name = data.get("name") or ""
                url = version_url(data)
                for file_obj in data.get("files") or []:
                    matched_type = same_hash(file_obj, hv)
                    if matched_type or label.startswith("Legacy"):
                        size_kb = file_obj.get("sizeKB") or 0
                        rows.append({
                            "hashQueried": label,
                            "hashMatched": matched_type or label,
                            "hashValue": hv,
                            "model": model_name,
                            "version": version_name,
                            "file": file_obj.get("name") or "",
                            "fileType": file_obj.get("type") or "",
                            "sizeMB": round(float(size_kb) / 1024, 2) if size_kb else "",
                            "modelId": model_id,
                            "versionId": version_id or "",
                            "url": url,
                            "rawFile": file_obj,
                        })
                time.sleep(0.25)

            self.q.put(("results", rows))
        except ssl.SSLCertVerificationError as e:
            self.q.put(("error", "SSL certificate verification failed. Try installing/updating certifi with:\n\npython -m pip install --upgrade certifi\n\nIf that still fails on your system, enable the 'Disable SSL verification' checkbox and retry.\n\nDetails: " + str(e)))
        except URLError as e:
            reason = getattr(e, "reason", e)
            if isinstance(reason, ssl.SSLCertVerificationError):
                self.q.put(("error", "SSL certificate verification failed. Try installing/updating certifi with:\n\npython -m pip install --upgrade certifi\n\nIf that still fails on your system, enable the 'Disable SSL verification' checkbox and retry.\n\nDetails: " + str(reason)))
            else:
                self.q.put(("error", str(e)))
        except Exception as e:
            self.q.put(("error", str(e)))

    def _poll_queue(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, read, total = msg
                    self.progress["maximum"] = max(total, 1)
                    self.progress["value"] = read
                    self.status.set(f"Hashing file... {read / max(total, 1) * 100:.1f}%")
                elif kind == "hashes":
                    _, sha, old = msg
                    self.sha256_value.set(sha)
                    self.old_hash_value.set(old)
                    self.status.set("Hash complete. Searching Civitai...")
                elif kind == "status":
                    self.status.set(msg[1])
                elif kind == "results":
                    rows = msg[1]
                    self.results = rows
                    for row in rows:
                        self.tree.insert("", "end", values=(
                            row["hashMatched"],
                            row["model"],
                            row["version"],
                            row["file"],
                            row["fileType"],
                            row["sizeMB"],
                            row["modelId"],
                            row["versionId"],
                            row["url"],
                        ))
                    self.match_count.set(str(len(rows)))
                    self.status.set(f"Done. Found {len(rows)} matching file record(s).")
                elif kind == "error":
                    self.status.set("Error.")
                    messagebox.showerror(APP_TITLE, msg[1])
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def export_json(self):
        if not self.results:
            messagebox.showinfo(APP_TITLE, "No results to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Save results",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        payload = {
            "selectedFile": self.selected_path.get(),
            "sha256": self.sha256_value.get(),
            "legacyHash": self.old_hash_value.get(),
            "matches": self.results,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        self.status.set(f"Exported JSON: {path}")


if __name__ == "__main__":
    app = HashComparerGUI()
    app.mainloop()
