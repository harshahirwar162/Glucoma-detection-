"""
Entry point for the fundus training pipeline.
Run this from the project root:
    python run_train.py --epochs 20 --batch_size 8 --val_split 0.2
"""
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from src.train import load_dataset_from_directory, train_model
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-input fundus model")
    parser.add_argument("--dataset_dir", type=str,
                        default=r"c:\Users\YUVRAJ\Downloads\tanavbajaj",
                        help="Path to the dataset directory")
    parser.add_argument("--epochs",     type=int,   default=20,   help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=16,   help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val_split",  type=float, default=0.2,  help="Validation fraction (0-1)")
    parser.add_argument("--save_path",  type=str,   default="best_model.pth",
                        help="Where to save the best checkpoint")
    parser.add_argument("--history_path", type=str, default="training_history.json",
                        help="Where to save training history JSON")
    args = parser.parse_args()

    print(f"Loading dataset from: {args.dataset_dir}")
    image_paths, labels = load_dataset_from_directory(args.dataset_dir)

    if not image_paths:
        print("No images found! Please check the dataset directory.")
    else:
        print(f"Found {len(image_paths)} images across {len(set(labels))} classes.")
        print("Starting training pipeline...")
        train_model(
            image_paths,
            labels,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_split=args.val_split,
            save_path=args.save_path,
            history_path=args.history_path,
        )
