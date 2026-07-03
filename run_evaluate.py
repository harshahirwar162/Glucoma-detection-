"""
run_evaluate.py — Top-level launcher for all research paper outputs.

Usage:
    # Step 1: Train and save model + history
    python run_train.py --epochs 20 --batch_size 16 --val_split 0.2

    # Step 2: Generate all research figures
    python run_evaluate.py
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from src.train import load_dataset_from_directory
from src.evaluate import run_full_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all research paper outputs")
    parser.add_argument("--dataset_dir",  type=str,   default=r"c:\Users\YUVRAJ\Downloads\tanavbajaj",
                        help="Path to the dataset directory")
    parser.add_argument("--model_path",   type=str,   default="best_model.pth",
                        help="Path to saved model weights (.pth)")
    parser.add_argument("--history_path", type=str,   default="training_history.json",
                        help="Path to training history JSON")
    parser.add_argument("--val_split",    type=float, default=0.2,
                        help="Must match the val_split used during training")
    parser.add_argument("--batch_size",   type=int,   default=16)
    args = parser.parse_args()

    print(f"Loading dataset from: {args.dataset_dir}")
    image_paths, labels = load_dataset_from_directory(args.dataset_dir)

    if not image_paths:
        print("No images found! Check --dataset_dir.")
        sys.exit(1)

    print(f"Found {len(image_paths)} images across {len(set(labels))} classes.")

    run_full_evaluation(
        image_paths=image_paths,
        labels=labels,
        model_path=args.model_path,
        history_path=args.history_path,
        val_split=args.val_split,
        batch_size=args.batch_size,
    )
