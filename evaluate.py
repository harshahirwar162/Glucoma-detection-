"""
evaluate.py — Generates all research paper figures and statistics.
Outputs go to the `outputs/` directory.
"""

import os
import json
import glob
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, cohen_kappa_score,
    matthews_corrcoef, accuracy_score
)
from sklearn.preprocessing import label_binarize

from .model import MultiBackboneFundusModel
from .dataset import FundusMultiInputDataset
from .preprocessing import preprocess_pipeline
from .gradcam_utils import generate_gradcam_overlay, prepare_input

# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = ['Normal', 'Early', 'Moderate', 'Deep', 'OHT']
CLASS_COLORS = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
PALETTE = sns.color_palette(CLASS_COLORS)
PLT_DPI  = 180
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=PLT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK]  Saved -> {path}")
    return path


def _val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _train_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# A. DATASET DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
def plot_dataset_distribution(image_paths, labels):
    print("\n[A] Dataset Distribution...")
    counts = [labels.count(i) for i in range(5)]
    total  = sum(counts)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Dataset Class Distribution", fontsize=14, fontweight='bold', y=1.01)

    # Bar chart
    bars = axes[0].bar(CLASS_NAMES, counts, color=CLASS_COLORS, edgecolor='white', linewidth=1.2, zorder=3)
    axes[0].set_title("Sample Count per Class", fontweight='bold')
    axes[0].set_ylabel("Number of Images")
    axes[0].set_xlabel("Glaucoma Stage")
    axes[0].grid(axis='y', alpha=0.4, zorder=0)
    axes[0].set_facecolor('#f8f9fa')
    for bar, cnt in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     str(cnt), ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Pie chart
    wedges, texts, autotexts = axes[1].pie(
        counts, labels=CLASS_NAMES, colors=CLASS_COLORS,
        autopct='%1.1f%%', startangle=140,
        wedgeprops=dict(edgecolor='white', linewidth=1.5))
    for at in autotexts:
        at.set_fontsize(9)
    axes[1].set_title(f"Class Proportions  (Total={total})", fontweight='bold')

    fig.tight_layout()
    return _save(fig, "dataset_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# B. TRAINING CURVES
# ═══════════════════════════════════════════════════════════════════════════════
def plot_training_curves(history_path="training_history.json"):
    print("\n[B] Training Curves...")
    if not os.path.exists(history_path):
        print(f"  ⚠  {history_path} not found — skip")
        return None

    with open(history_path) as f:
        h = json.load(f)

    epochs = range(1, len(h["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training History", fontsize=14, fontweight='bold')

    # Loss
    axes[0].plot(epochs, h["train_loss"], 'o-', color='#e74c3c', lw=2, ms=4, label='Train Loss')
    axes[0].plot(epochs, h["val_loss"],   's--', color='#3498db', lw=2, ms=4, label='Val Loss')
    axes[0].set_title("Loss per Epoch", fontweight='bold')
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_facecolor('#f8f9fa')

    # Accuracy
    axes[1].plot(epochs, h["train_acc"], 'o-', color='#e74c3c', lw=2, ms=4, label='Train Acc')
    axes[1].plot(epochs, h["val_acc"],   's--', color='#3498db', lw=2, ms=4, label='Val Acc')
    axes[1].set_title("Accuracy per Epoch (%)", fontweight='bold')
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_facecolor('#f8f9fa')

    # Annotations
    best_val_ep = int(np.argmax(h["val_acc"])) + 1
    best_val_acc = max(h["val_acc"])
    axes[1].axvline(best_val_ep, color='#27ae60', ls=':', lw=1.5,
                    label=f'Best Val ({best_val_acc:.1f}% @ ep {best_val_ep})')
    axes[1].legend()

    fig.tight_layout()
    return _save(fig, "training_curves.png")


# ═══════════════════════════════════════════════════════════════════════════════
# C. CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(all_labels, all_preds):
    print("\n[C] Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Confusion Matrix", fontsize=14, fontweight='bold')

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Absolute Counts", "Normalized (row %)"],
        ['d', '.2f']
    ):
        cmap = LinearSegmentedColormap.from_list('fundus', ['#ffffff', '#2c3e50'])
        sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    linewidths=0.5, linecolor='#ecf0f1', ax=ax,
                    cbar_kws={'shrink': 0.8})
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
        ax.tick_params(axis='x', rotation=30)
        ax.tick_params(axis='y', rotation=0)

    fig.tight_layout()
    return _save(fig, "confusion_matrix.png")


# ═══════════════════════════════════════════════════════════════════════════════
# D. ROC CURVES
# ═══════════════════════════════════════════════════════════════════════════════
def plot_roc_curves(all_labels, all_probs):
    print("\n[D] ROC Curves...")
    y_bin = label_binarize(all_labels, classes=list(range(5)))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('#f8f9fa')

    for i, (cname, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=color, label=f'{cname}  (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Classifier')
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (One-vs-Rest)', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, "roc_curves.png")


# ═══════════════════════════════════════════════════════════════════════════════
# E. METRICS REPORT
# ═══════════════════════════════════════════════════════════════════════════════
def compute_and_save_metrics(all_labels, all_preds, all_probs):
    print("\n[E] Metrics Report...")
    y_bin = label_binarize(all_labels, classes=list(range(5)))

    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4)
    kappa  = cohen_kappa_score(all_labels, all_preds)
    mcc    = matthews_corrcoef(all_labels, all_preds)
    acc    = accuracy_score(all_labels, all_preds)

    per_class_auc = {}
    for i, cname in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        per_class_auc[cname] = auc(fpr, tpr)

    lines = [
        "=" * 65,
        "   GLAUCOMA GRADING — EVALUATION METRICS REPORT",
        "=" * 65,
        "",
        f"  Overall Accuracy   : {acc*100:.2f}%",
        f"  Cohen's Kappa      : {kappa:.4f}",
        f"  Matthews CC (MCC)  : {mcc:.4f}",
        "",
        "  Per-Class AUC-ROC (One-vs-Rest):",
    ]
    for cname, a in per_class_auc.items():
        lines.append(f"    {cname:<12}: {a:.4f}")
    lines += [
        "",
        "-" * 65,
        "  Classification Report:",
        "-" * 65,
        report,
    ]

    text = "\n".join(lines)
    print(text)

    txt_path = os.path.join(OUTPUT_DIR, "metrics_report.txt")
    with open(txt_path, "w") as f:
        f.write(text)
    print(f"  [OK]  Saved -> {txt_path}")

    # Also save CSV-friendly row
    import csv
    csv_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Accuracy (%)", f"{acc*100:.4f}"])
        writer.writerow(["Cohen Kappa", f"{kappa:.4f}"])
        writer.writerow(["MCC", f"{mcc:.4f}"])
        for cname, a in per_class_auc.items():
            writer.writerow([f"AUC {cname}", f"{a:.4f}"])
    print(f"  [OK]  Saved -> {csv_path}")

    return acc, kappa, mcc, per_class_auc


# ═══════════════════════════════════════════════════════════════════════════════
# F. PREPROCESSING VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def plot_preprocessing_samples(image_paths, labels):
    print("\n[F] Preprocessing Visualization...")
    from .preprocessing import apply_clahe, apply_mca, extract_optic_disc_roi

    # Pick one sample per class
    samples = {}
    for path, lbl in zip(image_paths, labels):
        if lbl not in samples:
            samples[lbl] = path
        if len(samples) == 5:
            break

    stages = ['Original', 'CLAHE', 'MCA\n(Texture)', 'OD ROI']
    n_classes = len(samples)
    fig, axes = plt.subplots(n_classes, 4, figsize=(14, 3.2 * n_classes))
    fig.suptitle("Preprocessing Pipeline — One Sample per Class",
                 fontsize=13, fontweight='bold', y=1.01)

    for row, (lbl, path) in enumerate(sorted(samples.items())):
        img_bgr = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        clahe_img = apply_clahe(img_rgb)
        mca_img   = apply_mca(clahe_img)
        od_roi    = extract_optic_disc_roi(clahe_img)

        imgs_list = [img_rgb, clahe_img, mca_img, od_roi]

        for col, (stage_img, stage_name) in enumerate(zip(imgs_list, stages)):
            ax = axes[row][col] if n_classes > 1 else axes[col]
            ax.imshow(stage_img)
            ax.axis('off')
            if row == 0:
                ax.set_title(stage_name, fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(CLASS_NAMES[lbl], fontsize=10, fontweight='bold', rotation=90)
                ax.yaxis.set_label_coords(-0.05, 0.5)

    fig.tight_layout()
    return _save(fig, "preprocessing_samples.png")


# ═══════════════════════════════════════════════════════════════════════════════
# G. AUGMENTATION SAMPLES
# ═══════════════════════════════════════════════════════════════════════════════
def plot_augmentation_samples(image_paths):
    print("\n[G] Augmentation Samples...")
    aug_transform = _train_transform()
    # Inverse normalize for visualization
    inv_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    ])

    sample_path = image_paths[0]
    img_bgr = cv2.imread(sample_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    N = 8  # augmented versions
    fig, axes = plt.subplots(2, N // 2 + 1, figsize=(16, 6))
    fig.suptitle("Data Augmentation Samples (same source image)",
                 fontsize=13, fontweight='bold')

    # Original
    axes[0][0].imshow(img_rgb)
    axes[0][0].set_title("Original", fontweight='bold', fontsize=9)
    axes[0][0].axis('off')
    axes[1][0].axis('off')

    for i in range(N):
        row = i // (N // 2)
        col = i % (N // 2) + 1
        aug_tensor = aug_transform(pil_img)
        aug_vis = inv_normalize(aug_tensor).permute(1, 2, 0).numpy()
        aug_vis = np.clip(aug_vis, 0, 1)
        axes[row][col].imshow(aug_vis)
        axes[row][col].set_title(f"Aug #{i+1}", fontsize=9)
        axes[row][col].axis('off')

    fig.tight_layout()
    return _save(fig, "augmentation_samples.png")


# ═══════════════════════════════════════════════════════════════════════════════
# H. GRAD-CAM GRID
# ═══════════════════════════════════════════════════════════════════════════════
def plot_gradcam_grid(model, image_paths, labels, device):
    print("\n[H] Grad-CAM Grid...")
    transform = _val_transform()

    # Pick one sample per class
    samples = {}
    for path, lbl in zip(image_paths, labels):
        if lbl not in samples:
            samples[lbl] = path
        if len(samples) == 5:
            break

    target_effnet = model.branch1.features[-1]
    target_resnet = model.branch2.layer4[-1]

    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.5 * n))
    fig.suptitle("Grad-CAM Explanations — EfficientNet (Full) & ResNet50 (OD ROI)",
                 fontsize=12, fontweight='bold', y=1.01)

    col_titles = ['Original Image', 'EfficientNet Grad-CAM\n(Full Fundus)', 'ResNet50 Grad-CAM\n(Optic Disc ROI)']
    for col, ct in enumerate(col_titles):
        axes[0][col].set_title(ct, fontsize=9, fontweight='bold')

    model.eval()
    for row, (lbl, path) in enumerate(sorted(samples.items())):
        img_bgr = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        from .preprocessing import preprocess_pipeline
        full_proc, od_roi_proc = preprocess_pipeline(img_rgb)

        full_vis = cv2.resize(full_proc, (224, 224)).astype(np.float32) / 255.0
        od_vis   = cv2.resize(od_roi_proc, (224, 224)).astype(np.float32) / 255.0

        t_full = prepare_input(full_proc, transform, device)
        t_roi  = prepare_input(od_roi_proc, transform, device)
        input_pair = (t_full, t_roi)

        # EfficientNet CAM
        _, overlay_eff, pred = generate_gradcam_overlay(
            model, input_pair, target_effnet, full_vis, class_idx=lbl)

        # ResNet CAM
        _, overlay_res, _ = generate_gradcam_overlay(
            model, input_pair, target_resnet, od_vis, class_idx=lbl)

        axes[row][0].imshow(img_rgb)
        axes[row][0].set_ylabel(f"{CLASS_NAMES[lbl]}\n(pred: {CLASS_NAMES[pred]})",
                                 fontsize=8, fontweight='bold')
        axes[row][1].imshow(overlay_eff)
        axes[row][2].imshow(overlay_res)

        for col in range(3):
            axes[row][col].axis('off')

    fig.tight_layout()
    return _save(fig, "gradcam_grid.png")


# ═══════════════════════════════════════════════════════════════════════════════
# I. ARCHITECTURE DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════════
def plot_architecture_diagram():
    print("\n[I] Architecture Diagram...")

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9)
    ax.axis('off')
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    def box(text, x, y, w=2.2, h=0.7, fc='#16213e', ec='#0f3460', tc='white', fs=8, bold=False):
        rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                        boxstyle="round,pad=0.05",
                                        facecolor=fc, edgecolor=ec, linewidth=1.5)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x, y, text, ha='center', va='center', fontsize=fs,
                color=tc, fontweight=weight, wrap=True,
                multialignment='center')

    def arrow(x1, y1, x2, y2, color='#e0e0e0'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # Title
    ax.text(8, 8.5, 'Multi-Input Fundus Glaucoma Grading Pipeline',
            ha='center', va='center', fontsize=13, color='white', fontweight='bold')

    # ── Input block ──
    box('Fundus\nImage (RGB)', 2, 6.5, fc='#0f3460', ec='#e94560', tc='white', fs=8, bold=True)

    # ── Preprocessing ──
    box('CLAHE\nEnhancement', 4, 7.3, fc='#533483', ec='#a855f7', tc='white', fs=7.5)
    box('MCA\n(Texture)', 4, 6.5, fc='#533483', ec='#a855f7', tc='white', fs=7.5)
    box('OD ROI\nExtraction', 4, 5.7, fc='#533483', ec='#a855f7', tc='white', fs=7.5)

    arrow(2.55, 6.7, 3.9, 7.3)
    arrow(2.55, 6.5, 3.9, 6.5)
    arrow(2.55, 6.3, 3.9, 5.7)

    ax.text(4, 8.0, 'Preprocessing', ha='center', fontsize=9, color='#a855f7', fontweight='bold')

    # ── Branch 1: EfficientNet ──
    box('Full Processed\nImage (224×224)', 6.5, 7.3, fc='#1e3a5f', ec='#3498db', tc='white', fs=7.5)
    box('EfficientNet-B0\n(Pretrained)', 8.5, 7.3, fc='#1e3a5f', ec='#3498db', tc='white', fs=7.5)
    box('Features\n1280-dim', 10.4, 7.3, fc='#1e4080', ec='#5dade2', tc='white', fs=7.5)

    arrow(4.55, 7.3, 5.9, 7.3)
    arrow(7.6, 7.3, 7.95, 7.3)
    arrow(9.6, 7.3, 9.85, 7.3)

    ax.text(8.5, 8.0, 'Branch 1 — Full Image', ha='center', fontsize=9,
            color='#3498db', fontweight='bold')

    # ── Branch 2: ResNet ──
    box('OD ROI\n(224×224)', 6.5, 5.7, fc='#3d1a1a', ec='#e74c3c', tc='white', fs=7.5)
    box('ResNet-50\n(Pretrained)', 8.5, 5.7, fc='#3d1a1a', ec='#e74c3c', tc='white', fs=7.5)
    box('Features\n2048-dim', 10.4, 5.7, fc='#5a1e1e', ec='#ec7063', tc='white', fs=7.5)

    arrow(4.55, 5.7, 5.9, 5.7)
    arrow(7.6, 5.7, 7.95, 5.7)
    arrow(9.6, 5.7, 9.85, 5.7)

    ax.text(8.5, 5.0, 'Branch 2 — Optic Disc ROI', ha='center', fontsize=9,
            color='#e74c3c', fontweight='bold')

    # ── Fusion ──
    box('Adaptive Attention\nFusion Module\n(1024-dim)', 12.2, 6.5, w=2.6, h=1.2,
        fc='#1e4a2f', ec='#2ecc71', tc='white', fs=8, bold=True)

    arrow(11.5, 7.3, 11.55, 6.85)
    arrow(11.5, 5.7, 11.55, 6.15)

    # ── Classifier ──
    box('FC-512\nDropout(0.4)', 14.2, 6.7, fc='#2c3e50', ec='#f39c12', tc='white', fs=7.5)
    box('5-Class\nSoftmax Output', 14.2, 5.7, fc='#0f3460', ec='#f39c12', tc='white', fs=7.5,
        bold=True)

    arrow(13.55, 6.5, 13.65, 6.7)
    arrow(14.2, 6.35, 14.2, 6.05)

    ax.text(14.2, 7.4, 'Classifier', ha='center', fontsize=9, color='#f39c12', fontweight='bold')

    # ── Class labels ──
    class_y = [5.3, 5.1, 4.9, 4.7, 4.5]
    class_colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
    for i, (cn, cy, cc) in enumerate(zip(CLASS_NAMES, class_y, class_colors)):
        ax.annotate('', xy=(14.2, cy + 0.05), xytext=(14.2, 5.35),
                    arrowprops=dict(arrowstyle='->', color=cc, lw=1))
        ax.text(15.3, cy + 0.05, cn, ha='left', va='center', fontsize=7.5,
                color=cc, fontweight='bold')

    # ── Grad-CAM note ──
    box('Grad-CAM\nExplainability', 8.5, 4.0, fc='#2c2c54', ec='#706fd3', tc='white', fs=7.5)
    ax.annotate('', xy=(8.5, 4.35), xytext=(8.5, 5.35),
                arrowprops=dict(arrowstyle='<-', color='#706fd3', lw=1.2, linestyle='dashed'))

    fig.tight_layout(pad=0.5)
    return _save(fig, "architecture_diagram.png")


# ═══════════════════════════════════════════════════════════════════════════════
# J. ATTENTION WEIGHT DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
def plot_attention_weights(model, image_paths, labels, device):
    print("\n[J] Attention Weight Distribution...")
    transform = _val_transform()
    dataset = FundusMultiInputDataset(image_paths, labels, transform=transform)
    loader  = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    w1_all, w2_all, label_all = [], [], []

    # Hook to extract attention weights
    attn_weights = []
    def hook_fn(module, input, output):
        attn_weights.append(output.detach().cpu())

    handle = model.fusion.attention_fc[-1].register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for full_img, od_roi, lbl in loader:
            attn_weights.clear()
            model(full_img.to(device), od_roi.to(device))
            if attn_weights:
                w = attn_weights[0]  # (B, 2)
                w1_all.extend(w[:, 0].numpy().tolist())
                w2_all.extend(w[:, 1].numpy().tolist())
                label_all.extend(lbl.numpy().tolist())

    handle.remove()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Adaptive Attention Weights Distribution (w₁=EfficientNet, w₂=ResNet)",
                 fontsize=12, fontweight='bold')

    # Overall histogram
    axes[0].hist(w1_all, bins=20, alpha=0.7, color='#3498db', label='w₁ (EfficientNet)')
    axes[0].hist(w2_all, bins=20, alpha=0.7, color='#e74c3c', label='w₂ (ResNet)')
    axes[0].set_xlabel("Attention Weight Value"); axes[0].set_ylabel("Count")
    axes[0].set_title("Overall Distribution"); axes[0].legend()
    axes[0].grid(alpha=0.3); axes[0].set_facecolor('#f8f9fa')

    # Per-class boxplot
    data_w1 = [[] for _ in range(5)]
    data_w2 = [[] for _ in range(5)]
    for w1, w2, lbl in zip(w1_all, w2_all, label_all):
        data_w1[lbl].append(w1)
        data_w2[lbl].append(w2)

    positions = np.arange(5)
    bx1 = axes[1].boxplot(data_w1, positions=positions - 0.2, widths=0.3,
                           patch_artist=True, boxprops=dict(facecolor='#3498db', alpha=0.7))
    bx2 = axes[1].boxplot(data_w2, positions=positions + 0.2, widths=0.3,
                           patch_artist=True, boxprops=dict(facecolor='#e74c3c', alpha=0.7))
    axes[1].set_xticks(positions); axes[1].set_xticklabels(CLASS_NAMES, rotation=15)
    axes[1].set_xlabel("Class"); axes[1].set_ylabel("Attention Weight")
    axes[1].set_title("Per-Class Attention Weights")
    axes[1].legend([bx1['boxes'][0], bx2['boxes'][0]], ['w₁ EfficientNet', 'w₂ ResNet'],
                    loc='upper right')
    axes[1].grid(alpha=0.3); axes[1].set_facecolor('#f8f9fa')

    fig.tight_layout()
    return _save(fig, "attention_weights.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER RUNNER
# ═══════════════════════════════════════════════════════════════════════════════
def run_full_evaluation(
    image_paths, labels,
    model_path="best_model.pth",
    history_path="training_history.json",
    val_split=0.2,
    batch_size=16,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  FUNDUS RESEARCH EVALUATION  |  device={device}")
    print(f"{'='*60}")

    # ── Load model ──
    model = MultiBackboneFundusModel(num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"\n  Model loaded from {model_path}")

    # ── Build val split (same seed) ──
    transform = _val_transform()
    dataset = FundusMultiInputDataset(image_paths, labels, transform=transform)
    total = len(dataset)
    val_size  = max(1, int(total * val_split))
    train_size = total - val_size
    _, val_subset = random_split(dataset, [train_size, val_size],
                                  generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Inference ──
    print("\n  Running inference on validation set...")
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for full_img, od_roi, lbl in val_loader:
            outputs = model(full_img.to(device), od_roi.to(device))
            probs   = torch.softmax(outputs, dim=1).cpu().numpy()
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(lbl.numpy().tolist())

    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Generate all outputs ──
    plot_dataset_distribution(image_paths, labels)
    plot_training_curves(history_path)
    plot_confusion_matrix(all_labels, all_preds)
    plot_roc_curves(all_labels, all_probs)
    compute_and_save_metrics(all_labels, all_preds, all_probs)
    plot_preprocessing_samples(image_paths, labels)
    plot_augmentation_samples(image_paths)
    plot_gradcam_grid(model, image_paths, labels, device)
    plot_architecture_diagram()
    plot_attention_weights(model, image_paths, labels, device)

    print(f"\n{'='*60}")
    print(f"  All outputs saved in -> ./{OUTPUT_DIR}/")
    print(f"{'='*60}\n")
