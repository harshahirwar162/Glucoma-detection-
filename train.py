import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import glob
import argparse
import json

# Imports from local project
from .model import MultiBackboneFundusModel
from .dataset import FundusMultiInputDataset


def load_dataset_from_directory(directory_path: str):
    """Scan class subdirectories and return image paths + integer labels."""
    class_mapping = {
        'normal': 0,
        'Early': 1,
        'Moderate': 2,
        'Deep': 3,
        'OHT': 4,
    }

    image_paths = []
    labels = []

    for class_name, label in class_mapping.items():
        class_dir = os.path.join(directory_path, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Directory not found for class '{class_name}' at {class_dir}")
            continue

        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'):
            for img_path in glob.glob(os.path.join(class_dir, ext)):
                image_paths.append(img_path)
                labels.append(label)

    return image_paths, labels


def get_transforms():
    """Return (train_transform, val_transform)."""
    train_transform = transforms.Compose([
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def train_model(
    image_paths: list,
    labels: list,
    num_epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    val_split: float = 0.2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: str = "best_model.pth",
    history_path: str = "training_history.json",
):
    print(f"Using device: {device}")

    train_transform, val_transform = get_transforms()

    full_dataset = FundusMultiInputDataset(image_paths, labels, transform=train_transform)

    total = len(full_dataset)
    val_size = max(1, int(total * val_split))
    train_size = total - val_size

    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Override val subset with clean (no-augmentation) transforms
    val_subset.dataset = FundusMultiInputDataset(image_paths, labels, transform=val_transform)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=(device == "cuda"))

    print(f"Train samples: {train_size}  |  Val samples: {val_size}")

    model = MultiBackboneFundusModel(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_acc = 0.0

    # History tracking for research paper plots
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "epochs": num_epochs,
        "train_size": train_size,
        "val_size": val_size,
    }

    for epoch in range(num_epochs):
        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for full_img, od_roi, label in train_loader:
            full_img, od_roi, label = full_img.to(device), od_roi.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(full_img, od_roi)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * full_img.size(0)
            _, predicted   = outputs.max(1)
            train_total   += label.size(0)
            train_correct += predicted.eq(label).sum().item()

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for full_img, od_roi, label in val_loader:
                full_img, od_roi, label = full_img.to(device), od_roi.to(device), label.to(device)
                outputs = model(full_img, od_roi)
                loss = criterion(outputs, label)

                val_loss    += loss.item() * full_img.size(0)
                _, predicted = outputs.max(1)
                val_total   += label.size(0)
                val_correct += predicted.eq(label).sum().item()

        t_loss = train_loss / train_total
        t_acc  = 100. * train_correct / train_total
        v_loss = val_loss / val_total
        v_acc  = 100. * val_correct / val_total

        history["train_loss"].append(round(t_loss, 6))
        history["train_acc"].append(round(t_acc, 4))
        history["val_loss"].append(round(v_loss, 6))
        history["val_acc"].append(round(v_acc, 4))

        flag = " ✓ best" if v_acc > best_val_acc else ""
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), save_path)

        print(
            f"Epoch [{epoch+1:>2}/{num_epochs}]  "
            f"Train Loss: {t_loss:.4f}  Train Acc: {t_acc:.1f}%  |  "
            f"Val Loss: {v_loss:.4f}  Val Acc: {v_acc:.1f}%{flag}"
        )

    # Save history for research paper plots
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved → {history_path}")
    print(f"Training complete. Best val acc: {best_val_acc:.1f}%  (saved → {save_path})")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-input fundus model")
    parser.add_argument("--dataset_dir", type=str, default="c:/Users/YUVRAJ/Downloads/tanavbajaj")
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--val_split",   type=float, default=0.2)
    parser.add_argument("--save_path",   type=str,   default="best_model.pth")
    parser.add_argument("--history_path",type=str,   default="training_history.json")
    args = parser.parse_args()

    image_paths, labels = load_dataset_from_directory(args.dataset_dir)
    if not image_paths:
        print("No images found!")
    else:
        print(f"Found {len(image_paths)} images across {len(set(labels))} classes.")
        train_model(
            image_paths, labels,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_split=args.val_split,
            save_path=args.save_path,
            history_path=args.history_path,
        )
