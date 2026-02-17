import json
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
from pathlib import Path
import os
from collections import defaultdict
import colorsys

class ConfigVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Training Config Visualizer - Pro Edition")
        self.root.geometry("700x500")
        self.config = None
        self.config_path = None
        self.dataset_stats = None
        
        # Professional color palette
        self.colors = {
            'primary': '#2E86AB',      # Professional blue
            'secondary': '#A23B72',    # Accent magenta
            'tertiary': '#F18F01',     # Warning orange
            'quaternary': '#C73E1D',   # Deep red
            'grid': '#E5E5E5',         # Light gray
            'text': '#333333',         # Dark gray
            'bg': '#FAFAFA'            # Off-white
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        # File selection
        file_frame = tk.Frame(self.root, padx=20, pady=20)
        file_frame.pack(fill=tk.X)
        
        tk.Label(file_frame, text="Config File:", font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W)
        
        self.path_var = tk.StringVar(value="No file selected")
        tk.Label(file_frame, textvariable=self.path_var, wraplength=650, fg='gray').pack(anchor=tk.W, pady=(5, 10))
        
        tk.Button(file_frame, text="Select Config JSON", command=self.load_config, 
                 bg=self.colors['primary'], fg='white', padx=20, pady=5).pack(anchor=tk.W)
        
        # Stats frame
        self.stats_frame = tk.LabelFrame(self.root, text="Dataset Statistics", padx=20, pady=10)
        self.stats_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.stats_text = tk.Text(self.stats_frame, height=8, wrap=tk.WORD, font=('Consolas', 9))
        self.stats_text.pack(fill=tk.X)
        self.stats_text.insert(tk.END, "Load a config file to scan datasets...")
        self.stats_text.config(state=tk.DISABLED)
        
        # Buttons frame
        btn_frame = tk.Frame(self.root, padx=20, pady=10)
        btn_frame.pack(fill=tk.X)
        
        buttons = [
            ("1. Learning Rate Schedule", self.plot_lr_schedule, self.colors['primary']),
            ("2. Timestep Allocation (Raw)", self.plot_timesteps_raw, self.colors['secondary']),
            ("3. Timestep Distribution with Shift", self.plot_timesteps_shifted, self.colors['tertiary']),
            ("4. Dataset Composition (Anonymous)", self.plot_dataset_composition, self.colors['quaternary']),
            ("5. Combined Dashboard", self.plot_dashboard, '#1B4965')
        ]
        
        self.buttons = []
        for text, cmd, color in buttons:
            btn = tk.Button(btn_frame, text=text, command=cmd, state=tk.DISABLED,
                          bg=color, fg='white', pady=5)
            btn.pack(fill=tk.X, pady=2)
            self.buttons.append(btn)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_config(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not file_path:
            return
            
        try:
            with open(file_path, 'r') as f:
                self.config = json.load(f)
            self.config_path = file_path
            
            # Scan datasets
            self.dataset_stats = self._scan_datasets()
            
            self.path_var.set(file_path)
            self._update_stats_display()
            
            for btn in self.buttons:
                btn.config(state=tk.NORMAL)
                
            self.status_var.set(f"Loaded: {Path(file_path).name} | Datasets scanned")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config:\n{str(e)}")
    
    def _scan_datasets(self):
        """Scan all dataset folders and count images recursively"""
        stats = {
            'total_images': 0,
            'datasets': [],
            'grand_total': 0
        }
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.gif'}
        
        for idx, dataset in enumerate(self.config.get('INSTANCE_DATASETS', [])):
            path = dataset.get('path', '')
            repeats = dataset.get('repeats', 1)
            
            if not os.path.exists(path):
                continue
                
            dataset_name = Path(path).name
            folder_counts = defaultdict(int)
            
            # Walk directory
            for root, dirs, files in os.walk(path):
                rel_path = Path(root).relative_to(path)
                folder_name = str(rel_path) if str(rel_path) != '.' else '(root)'
                
                # Count images
                count = sum(1 for f in files if Path(f).suffix.lower() in image_extensions)
                if count > 0:
                    folder_counts[folder_name] = count * repeats
            
            # Sort alphabetically
            sorted_folders = dict(sorted(folder_counts.items()))
            total = sum(sorted_folders.values())
            
            stats['datasets'].append({
                'name': dataset_name,
                'path': path,
                'folders': sorted_folders,
                'total': total,
                'repeats': repeats
            })
            stats['grand_total'] += total
        
        return stats
    
    def _update_stats_display(self):
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        
        if not self.dataset_stats or not self.dataset_stats['datasets']:
            self.stats_text.insert(tk.END, "No datasets found or invalid paths")
            self.stats_text.config(state=tk.DISABLED)
            return
        
        lines = [f"Grand Total: {self.dataset_stats['grand_total']:,} images\n"]
        
        for ds in self.dataset_stats['datasets']:
            pct = (ds['total'] / self.dataset_stats['grand_total']) * 100
            lines.append(f"\n{ds['name']}: {ds['total']:,} images ({pct:.1f}%) [repeats: {ds['repeats']}x]")
            lines.append("-" * 40)
            
            # Show only counts, not names, for privacy in the text view too
            counts = list(ds['folders'].values())
            lines.append(f"  Folders: {len(counts)} | Min: {min(counts)} | Max: {max(counts)} | Avg: {sum(counts)//len(counts)}")
        
        self.stats_text.insert(tk.END, '\n'.join(lines))
        self.stats_text.config(state=tk.DISABLED)
    
    def get_lr_at_step(self, step, total_steps, curve_points):
        """Interpolate LR from custom curve points"""
        progress = step / total_steps
        points = sorted(curve_points, key=lambda x: x[0])
        
        for i in range(len(points) - 1):
            if points[i][0] <= progress <= points[i+1][0]:
                t = (progress - points[i][0]) / (points[i+1][0] - points[i][0])
                lr = points[i][1] + t * (points[i+1][1] - points[i][1])
                return lr
        
        return points[-1][1] if progress >= points[-1][0] else points[0][1]
    
    def apply_flow_shift(self, t_normalized, shift_factor):
        """SD3/Flux schedule warping equation"""
        if shift_factor == 1.0 or shift_factor is None:
            return t_normalized
        return (shift_factor * t_normalized) / (1.0 + (shift_factor - 1.0) * t_normalized)
    
    def plot_lr_schedule(self):
        if not self.config:
            return
            
        try:
            total_steps = int(self.config.get('MAX_TRAIN_STEPS', 65090))
            curve_points = self.config.get('LR_CUSTOM_CURVE', [])
            
            steps = np.linspace(0, total_steps, 1000)
            lrs = [self.get_lr_at_step(s, total_steps, curve_points) for s in steps]
            
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
            ax.set_facecolor('white')
            
            # Plot curve
            ax.plot(steps, lrs, color=self.colors['primary'], linewidth=2.5, label='Learning Rate')
            ax.fill_between(steps, lrs, alpha=0.2, color=self.colors['primary'])
            
            # Mark control points
            for point in curve_points:
                step = point[0] * total_steps
                lr = point[1]
                ax.plot(step, lr, 'o', color=self.colors['secondary'], markersize=8, 
                       markeredgecolor='white', markeredgewidth=2, zorder=5)
                ax.annotate(f'{lr:.2e}', (step, lr), textcoords="offset points", 
                           xytext=(0,12), ha='center', fontsize=8, color=self.colors['text'])
            
            # Formatting
            ax.set_xlabel('Training Step', fontsize=11, fontweight='bold', color=self.colors['text'])
            ax.set_ylabel('Learning Rate', fontsize=11, fontweight='bold', color=self.colors['text'])
            ax.set_title('Learning Rate Schedule\n(Custom Curve with Interpolation)', 
                        fontsize=13, fontweight='bold', pad=20, color=self.colors['text'])
            ax.grid(True, alpha=0.3, color=self.colors['grid'])
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            
            # Add phase annotations
            peak_lr = max([p[1] for p in curve_points])
            ax.axhline(y=peak_lr, color=self.colors['tertiary'], linestyle='--', alpha=0.5, 
                      label=f'Peak LR: {peak_lr:.2e}')
            ax.legend(loc='upper right')
            
            plt.tight_layout()
            self.save_or_show(fig, "lr_schedule.png")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate LR plot:\n{str(e)}")
    
    def plot_timesteps_raw(self):
        if not self.config or 'TIMESTEP_ALLOCATION' not in self.config:
            messagebox.showwarning("Warning", "No timestep allocation data found in config")
            return
            
        try:
            alloc = self.config['TIMESTEP_ALLOCATION']
            bin_size = alloc['bin_size']
            counts = alloc['counts']
            
            bins = np.arange(len(counts)) * bin_size + bin_size/2
            
            fig, ax = plt.subplots(figsize=(12, 5), facecolor='white')
            ax.set_facecolor('white')
            
            bars = ax.bar(bins, counts, width=bin_size*0.9, color=self.colors['primary'], 
                         edgecolor='white', linewidth=0.5, alpha=0.8)
            
            # Highlight regions
            for i, bar in enumerate(bars):
                if i < 5:
                    bar.set_color('#FF6B6B')
                elif i > 19:
                    bar.set_color('#4ECDC4')
            
            ax.set_xlabel('Timestep (0-1000)', fontsize=11, fontweight='bold', color=self.colors['text'])
            ax.set_ylabel('Sample Count', fontsize=11, fontweight='bold', color=self.colors['text'])
            ax.set_title(f'Timestep Allocation (Raw)\nMode: {self.config.get("TIMESTEP_MODE", "Unknown")} | Bin Size: {bin_size}', 
                        fontsize=13, fontweight='bold', pad=20, color=self.colors['text'])
            ax.grid(True, axis='y', alpha=0.3, color=self.colors['grid'])
            
            # Legend
            red_patch = mpatches.Patch(color='#FF6B6B', label='High Noise (Early)')
            blue_patch = mpatches.Patch(color=self.colors['primary'], label='Mid Noise')
            teal_patch = mpatches.Patch(color='#4ECDC4', label='Low Noise (Late/Detail)')
            ax.legend(handles=[red_patch, blue_patch, teal_patch], loc='upper right')
            
            plt.tight_layout()
            self.save_or_show(fig, "timesteps_raw.png")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate timestep plot:\n{str(e)}")
    
    def plot_timesteps_shifted(self):
        if not self.config or 'TIMESTEP_ALLOCATION' not in self.config:
            messagebox.showwarning("Warning", "No timestep allocation data found")
            return
            
        try:
            alloc = self.config['TIMESTEP_ALLOCATION']
            bin_size = alloc['bin_size']
            counts = alloc['counts']
            shift_factor = self.config.get('RF_SHIFT_FACTOR', 3.0)
            
            original_bins = np.arange(len(counts)) * bin_size + bin_size/2
            original_norm = original_bins / 1000.0
            shifted_norm = self.apply_flow_shift(original_norm, shift_factor)
            shifted_bins = shifted_norm * 1000
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
            
            # Left: Mapping curve
            ax1.set_facecolor('white')
            t_all = np.linspace(0, 1000, 100)
            t_norm = t_all / 1000
            shifted_all = self.apply_flow_shift(t_norm, shift_factor) * 1000
            
            ax1.plot(t_all, shifted_all, color=self.colors['secondary'], linewidth=2.5, 
                    label=f'Shift Factor: {shift_factor}')
            ax1.plot([0, 1000], [0, 1000], '--', color='gray', alpha=0.5, label='No Shift (1.0)')
            
            ax1.axvspan(0, 200, alpha=0.2, color='red', label='High Noise → Compressed')
            ax1.axvspan(800, 1000, alpha=0.2, color='teal', label='Low Noise → Expanded')
            
            ax1.set_xlabel('Original Timestep', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Shifted Timestep', fontsize=11, fontweight='bold')
            ax1.set_title('Flow Shift Mapping\n(SD3/Flux Rectified Flow)', fontsize=12, fontweight='bold', pad=15)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left', fontsize=9)
            ax1.set_xlim(0, 1000)
            ax1.set_ylim(0, 1000)
            
            # Right: Shifted distribution
            ax2.set_facecolor('white')
            
            ax2.bar(shifted_bins, counts, width=bin_size*0.8, color=self.colors['tertiary'], 
                   edgecolor='white', alpha=0.7, label=f'Shifted (x{shift_factor})')
            ax2.bar(original_bins, counts, width=bin_size*0.4, color=self.colors['primary'], 
                   edgecolor='white', alpha=0.3, label='Original')
            
            ax2.set_xlabel('Effective Timestep', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Sample Count', fontsize=11, fontweight='bold')
            ax2.set_title(f'Timestep Distribution After Shift\n(Comparison: Original vs Shifted)', 
                         fontsize=12, fontweight='bold', pad=15)
            ax2.grid(True, axis='y', alpha=0.3)
            ax2.legend(loc='upper right')
            
            plt.tight_layout()
            self.save_or_show(fig, "timesteps_shifted.png")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate shifted plot:\n{str(e)}")
    
    def plot_dataset_composition(self):
        """Create anonymized hierarchical visualization by size tiers"""
        if not self.dataset_stats or not self.dataset_stats['datasets']:
            messagebox.showwarning("Warning", "No dataset statistics available")
            return
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), facecolor='white',
                                           gridspec_kw={'width_ratios': [1, 2]})
            
            # LEFT: Macro pie chart
            ax1.set_facecolor('white')
            dataset_names = [ds['name'] for ds in self.dataset_stats['datasets']]
            dataset_totals = [ds['total'] for ds in self.dataset_stats['datasets']]
            dataset_pcts = [(t/self.dataset_stats['grand_total'])*100 for t in dataset_totals]
            
            macro_colors = [self.colors['primary'], self.colors['secondary']]
            
            wedges, texts = ax1.pie(
                dataset_totals, 
                labels=[f"{name}\n{total:,} img\n({pct:.1f}%)" 
                       for name, total, pct in zip(dataset_names, dataset_totals, dataset_pcts)],
                colors=macro_colors,
                startangle=90,
                textprops={'fontsize': 10, 'fontweight': 'bold'}
            )
            ax1.set_title('Dataset Distribution\n(Macro Split)', fontsize=13, fontweight='bold', pad=20)
            
            # RIGHT: Anonymous tiered distribution
            ax2.set_facecolor('white')
            
            # Define size tiers
            tiers = {
                'XL (500+ img)': {'range': (500, 9999), 'color': '#D32F2F', 'count': 0, 'total': 0},
                'Large (200-499)': {'range': (200, 499), 'color': '#F57C00', 'count': 0, 'total': 0},
                'Medium (100-199)': {'range': (100, 199), 'color': '#FBC02D', 'count': 0, 'total': 0},
                'Small (50-99)': {'range': (50, 99), 'color': '#388E3C', 'count': 0, 'total': 0},
                'Tiny (<50)': {'range': (0, 49), 'color': '#7B1FA2', 'count': 0, 'total': 0}
            }
            
            # Categorize all folders
            tier_data = {name: [] for name in tiers.keys()}
            
            for ds in self.dataset_stats['datasets']:
                for folder, count in ds['folders'].items():
                    for tier_name, tier_info in tiers.items():
                        if tier_info['range'][0] <= count <= tier_info['range'][1]:
                            tier_data[tier_name].append(count)
                            tier_info['count'] += 1
                            tier_info['total'] += count
                            break
            
            # Create horizontal bar chart by tier
            y_pos = 0
            y_positions = []
            y_labels = []
            bar_colors = []
            bar_values = []
            
            for tier_name, tier_info in tiers.items():
                if tier_info['count'] > 0:
                    y_positions.append(y_pos)
                    y_labels.append(f"{tier_name}\n({tier_info['count']} sources)")
                    bar_colors.append(tier_info['color'])
                    bar_values.append(tier_info['total'])
                    
                    # Scatter points for individual folders
                    individual_counts = tier_data[tier_name]
                    jitter = np.random.uniform(-0.2, 0.2, len(individual_counts))
                    y_scatter = [y_pos + j for j in jitter]
                    
                    ax2.scatter(individual_counts, y_scatter, 
                               color=tier_info['color'], alpha=0.6, s=50, zorder=5)
                    
                    y_pos += 1
            
            # Draw bars
            bars = ax2.barh(y_positions, bar_values, color=bar_colors, 
                           edgecolor='white', linewidth=2, height=0.6, alpha=0.8)
            
            # Add value labels
            for bar, val, y in zip(bars, bar_values, y_positions):
                pct = (val / self.dataset_stats['grand_total']) * 100
                ax2.text(val + max(bar_values)*0.02, y, 
                        f"{val:,} img ({pct:.1f}%)", 
                        va='center', fontsize=10, fontweight='bold')
            
            # Legend
            legend_elements = [mpatches.Patch(color=tiers[t]['color'], label=t) 
                              for t in tiers.keys() if tiers[t]['count'] > 0]
            ax2.legend(handles=legend_elements, loc='lower right', 
                      title='Source Size Tiers', framealpha=0.9)
            
            ax2.set_yticks(y_positions)
            ax2.set_yticklabels(y_labels, fontsize=10)
            ax2.set_xlabel('Total Images in Tier', fontsize=11, fontweight='bold')
            ax2.set_title('Distribution by Source Size\n(Anonymous)', fontsize=13, fontweight='bold', pad=20)
            ax2.grid(True, axis='x', alpha=0.3, color=self.colors['grid'])
            
            # Warning about tiny sources
            tiny_count = tiers['Tiny (<50)']['count'] if tiers['Tiny (<50)']['count'] > 0 else 0
            if tiny_count > 0:
                ax2.text(0.02, 0.02, 
                        f"⚠️ {tiny_count} tiny sources (<50 img) at risk of overfitting\nConsider increasing repeats for these",
                        transform=ax2.transAxes, fontsize=9, color='red',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
            
            plt.tight_layout()
            self.save_or_show(fig, "dataset_anonymous.png")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate plot:\n{str(e)}")
    
    def plot_dashboard(self):
        """Create combined view of all metrics"""
        if not self.config:
            return
            
        try:
            fig = plt.figure(figsize=(16, 12), facecolor='white')
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
            
            # 1. LR Schedule (top left)
            ax_lr = fig.add_subplot(gs[0, 0])
            self._plot_lr_on_axis(ax_lr)
            ax_lr.set_title('Learning Rate Schedule', fontsize=11, fontweight='bold')
            
            # 2. Timestep Raw (top right)
            ax_raw = fig.add_subplot(gs[0, 1])
            self._plot_timesteps_on_axis(ax_raw, shifted=False)
            ax_raw.set_title('Raw Timestep Distribution', fontsize=11, fontweight='bold')
            
            # 3. Timestep Shifted (middle left)
            ax_shift = fig.add_subplot(gs[1, 0])
            self._plot_timesteps_on_axis(ax_shift, shifted=True)
            ax_shift.set_title('With Flow Shift (Factor: {})'.format(
                self.config.get('RF_SHIFT_FACTOR', 'N/A')), fontsize=11, fontweight='bold')
            
            # 4. Dataset Composition (middle right)
            ax_data = fig.add_subplot(gs[1, 1])
            self._plot_dataset_tiers_on_axis(ax_data)
            ax_data.set_title('Dataset Composition by Size', fontsize=11, fontweight='bold')
            
            # 5. Info table (bottom, full width)
            ax_info = fig.add_subplot(gs[2, :])
            ax_info.axis('off')
            
            # Info text
            info_lines = [
                f"Base Model: {Path(self.config.get('SINGLE_FILE_CHECKPOINT_PATH', 'Unknown')).name}",
                f"Training: {self.config.get('MAX_TRAIN_STEPS', 'N/A')} steps | "
                f"Batch: {self.config.get('BATCH_SIZE', 1)}×{self.config.get('GRADIENT_ACCUMULATION_STEPS', 1)} "
                f"= {int(self.config.get('BATCH_SIZE', 1))*int(self.config.get('GRADIENT_ACCUMULATION_STEPS', 1))} effective",
                f"LR: {self.config.get('LR_GRAPH_MIN', 'N/A')} → {self.config.get('LR_GRAPH_MAX', 'N/A')} | "
                f"Optimizer: {self.config.get('OPTIMIZER_TYPE', 'N/A')}",
                f"Shift: {self.config.get('RF_SHIFT_FACTOR', 'N/A')} | "
                f"Loss: {self.config.get('LOSS_TYPE', 'N/A')} (detail: {self.config.get('SEMANTIC_DETAIL_WEIGHT', 'N/A')}×)",
                f"Dropout: {float(self.config.get('UNCONDITIONAL_DROPOUT_CHANCE', 0))*100:.0f}% | "
                f"Precision: {self.config.get('MIXED_PRECISION', 'N/A')}",
                f"Total Dataset: {self.dataset_stats['grand_total']:,} images | "
                f"Sources: {sum(len(ds['folders']) for ds in self.dataset_stats['datasets'])} folders"
            ]
            
            table_text = '\n'.join([f'• {line}' for line in info_lines])
            ax_info.text(0.5, 0.5, table_text, transform=ax_info.transAxes, fontsize=10,
                        verticalalignment='center', horizontalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                        family='monospace')
            
            fig.suptitle(f'Training Configuration Dashboard\n{Path(self.config_path).name}', 
                        fontsize=14, fontweight='bold', y=0.98)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            self.save_or_show(fig, "training_dashboard.png")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate dashboard:\n{str(e)}")
    
    # Helper methods for dashboard
    def _plot_lr_on_axis(self, ax):
        total_steps = int(self.config.get('MAX_TRAIN_STEPS', 65090))
        curve_points = self.config.get('LR_CUSTOM_CURVE', [])
        steps = np.linspace(0, total_steps, 500)
        lrs = [self.get_lr_at_step(s, total_steps, curve_points) for s in steps]
        
        ax.plot(steps, lrs, color=self.colors['primary'], linewidth=2)
        ax.fill_between(steps, lrs, alpha=0.2, color=self.colors['primary'])
        ax.set_xlabel('Step', fontsize=9)
        ax.set_ylabel('LR', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    def _plot_timesteps_on_axis(self, ax, shifted=False):
        alloc = self.config['TIMESTEP_ALLOCATION']
        bin_size = alloc['bin_size']
        counts = alloc['counts']
        bins = np.arange(len(counts)) * bin_size + bin_size/2
        
        if shifted:
            shift_factor = self.config.get('RF_SHIFT_FACTOR', 3.0)
            bins_norm = bins / 1000.0
            bins = self.apply_flow_shift(bins_norm, shift_factor) * 1000
            color = self.colors['tertiary']
        else:
            color = self.colors['primary']
            
        ax.bar(bins, counts, width=bin_size*0.8, color=color, edgecolor='white', alpha=0.8)
        ax.set_xlabel('Timestep', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)
    
    def _plot_dataset_tiers_on_axis(self, ax):
        """Simplified tier plot for dashboard"""
        tiers = {
            'XL (500+)': {'range': (500, 9999), 'color': '#D32F2F'},
            'Large (200-499)': {'range': (200, 499), 'color': '#F57C00'},
            'Medium (100-199)': {'range': (100, 199), 'color': '#FBC02D'},
            'Small (50-99)': {'range': (50, 99), 'color': '#388E3C'},
            'Tiny (<50)': {'range': (0, 49), 'color': '#7B1FA2'}
        }
        
        tier_counts = defaultdict(int)
        for ds in self.dataset_stats['datasets']:
            for folder, count in ds['folders'].items():
                for tier_name, tier_info in tiers.items():
                    if tier_info['range'][0] <= count <= tier_info['range'][1]:
                        tier_counts[tier_name] += count
                        break
        
        labels = list(tier_counts.keys())
        values = list(tier_counts.values())
        colors = [tiers[l]['color'] for l in labels]
        
        ax.pie(values, labels=[f"{l}\n{v:,}" for l, v in zip(labels, values)], 
              colors=colors, startangle=90, textprops={'fontsize': 8})
        ax.set_title('Images by Source Size', fontsize=11, fontweight='bold')
    
    def save_or_show(self, fig, default_name):
        choice = messagebox.askyesnocancel("Save", f"Save chart as {default_name}?\n(Yes=Save, No=Show, Cancel=Abort)")
        if choice is True:
            save_path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=default_name)
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                self.status_var.set(f"Saved: {save_path}")
                plt.close(fig)
            else:
                plt.show()
        elif choice is False:
            plt.show()
        else:
            plt.close(fig)

if __name__ == "__main__":
    root = tk.Tk()
    app = ConfigVisualizer(root)
    root.mainloop()