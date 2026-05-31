#!/usr/bin/env python
"""Convert tag-style .txt caption sidecars to JSON caption sidecars.

The trainer's JSON caption mode looks for a .json file next to each image.
This script reads image-matched .txt sidecars and writes:

{
  "tags": "tag1, tag2, tag3"
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def image_for_caption(txt_path: Path) -> Path | None:
    for suffix in IMAGE_EXTENSIONS:
        candidate = txt_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def convert_txt_file(
    txt_path: Path,
    *,
    overwrite: bool,
    remove_txt: bool,
    dry_run: bool,
) -> str:
    image_path = image_for_caption(txt_path)
    if image_path is None:
        return "no_image"

    caption = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not caption:
        return "empty"

    json_path = txt_path.with_suffix(".json")
    if json_path.exists() and not overwrite:
        return "exists"

    if not dry_run:
        payload = {"tags": caption}
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        if remove_txt:
            txt_path.unlink()

    return "converted"


def iter_txt_files(paths: list[Path]) -> list[Path]:
    txt_files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() == ".txt":
            txt_files.append(path)
        elif path.is_dir():
            txt_files.extend(path.rglob("*.txt"))
    return sorted(set(txt_files))


def convert_paths(
    paths: list[Path],
    *,
    overwrite: bool = False,
    remove_txt: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    counts = {
        "converted": 0,
        "exists": 0,
        "empty": 0,
        "no_image": 0,
        "scanned": 0,
    }

    txt_files = iter_txt_files(paths)
    counts["scanned"] = len(txt_files)
    for txt_path in txt_files:
        result = convert_txt_file(
            txt_path,
            overwrite=overwrite,
            remove_txt=remove_txt,
            dry_run=dry_run,
        )
        counts[result] += 1
    return counts


class ConverterApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("TXT Caption to JSON Converter")
        self.geometry("560x300")
        self.minsize(520, 280)

        self.folder_var = tk.StringVar()
        self.overwrite_var = tk.BooleanVar(value=False)
        self.remove_txt_var = tk.BooleanVar(value=False)
        self.dry_run_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Select a dataset folder to begin.")

        self._build_ui()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)

        folder_row = ttk.Frame(root)
        folder_row.pack(fill="x")

        ttk.Label(folder_row, text="Dataset folder").pack(anchor="w")
        entry_row = ttk.Frame(folder_row)
        entry_row.pack(fill="x", pady=(4, 0))
        ttk.Entry(entry_row, textvariable=self.folder_var).pack(side="left", fill="x", expand=True)
        ttk.Button(entry_row, text="Browse...", command=self.browse_folder).pack(side="left", padx=(8, 0))

        options = ttk.LabelFrame(root, text="Options", padding=10)
        options.pack(fill="x", pady=(14, 0))
        ttk.Checkbutton(options, text="Overwrite existing .json files", variable=self.overwrite_var).pack(anchor="w")
        ttk.Checkbutton(options, text="Delete .txt files after conversion", variable=self.remove_txt_var).pack(anchor="w", pady=(4, 0))
        ttk.Checkbutton(options, text="Dry run only", variable=self.dry_run_var).pack(anchor="w", pady=(4, 0))

        button_row = ttk.Frame(root)
        button_row.pack(fill="x", pady=(14, 0))
        ttk.Button(button_row, text="Start", command=self.start_conversion).pack(side="right")

        status = ttk.Label(root, textvariable=self.status_var, justify="left", anchor="nw")
        status.pack(fill="both", expand=True, pady=(14, 0))

    def browse_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select Dataset Folder")
        if folder:
            self.folder_var.set(folder)

    def start_conversion(self) -> None:
        folder = Path(self.folder_var.get().strip())
        if not folder:
            messagebox.showwarning("No Folder", "Select a dataset folder first.")
            return
        if not folder.exists() or not folder.is_dir():
            messagebox.showerror("Invalid Folder", f"Folder not found:\n{folder}")
            return

        try:
            counts = convert_paths(
                [folder],
                overwrite=self.overwrite_var.get(),
                remove_txt=self.remove_txt_var.get(),
                dry_run=self.dry_run_var.get(),
            )
        except Exception as exc:
            messagebox.showerror("Conversion Failed", str(exc))
            return

        action = "Would convert" if self.dry_run_var.get() else "Converted"
        summary = (
            f"{action}: {counts['converted']}\n"
            f"Skipped existing JSON: {counts['exists']}\n"
            f"Skipped empty TXT: {counts['empty']}\n"
            f"Skipped TXT without matching image: {counts['no_image']}\n"
            f"Scanned TXT files: {counts['scanned']}"
        )
        self.status_var.set(summary)
        messagebox.showinfo("Done", summary)


def run_gui() -> int:
    app = ConverterApp()
    app.mainloop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert image-matched .txt captions into .json captions with a tags key."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Dataset folder(s) or individual .txt caption file(s) to convert. If omitted, opens the GUI.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing .json sidecars.",
    )
    parser.add_argument(
        "--remove-txt",
        action="store_true",
        help="Delete each .txt sidecar after its .json sidecar is written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would happen without writing or deleting files.",
    )
    args = parser.parse_args()

    if not args.paths:
        return run_gui()

    counts = convert_paths(
        args.paths,
        overwrite=args.overwrite,
        remove_txt=args.remove_txt,
        dry_run=args.dry_run,
    )

    action = "Would convert" if args.dry_run else "Converted"
    print(f"{action}: {counts['converted']}")
    print(f"Skipped existing JSON: {counts['exists']}")
    print(f"Skipped empty TXT: {counts['empty']}")
    print(f"Skipped TXT without matching image: {counts['no_image']}")
    print(f"Scanned TXT files: {counts['scanned']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
