import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple, List, Callable
from .preprocessing import preprocess_pipeline

class FundusMultiInputDataset(Dataset):
    """
    Dataset to yield (Full_Image, Optic_Disc_ROI, Label) tuples.
    """
    def __init__(self, 
                 image_paths: List[str], 
                 labels: List[int], 
                 transform: Callable = None):
        """
        Args:
            image_paths (list): List of paths to fundus images.
            labels (list): List of integer labels (0-4 for 5-class).
            transform (callable, optional): PyTorch transforms to apply to the resulting images.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image via OpenCV (RGB)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing pipeline
        full_img, od_roi = preprocess_pipeline(image)
        
        # Convert to PIL for typical torchvision transform compatibility
        full_img_pil = Image.fromarray(full_img)
        od_roi_pil = Image.fromarray(od_roi)
        
        if self.transform:
            # Note: Transform should ideally handle the resize to target dims (e.g., 224x224)
            # and ToTensor() / Normalization
            full_img_tensor = self.transform(full_img_pil)
            od_roi_tensor = self.transform(od_roi_pil)
        else:
            # Fallback tensor conversion if no transform provided
            import torchvision.transforms.functional as F
            full_img_tensor = F.to_tensor(full_img_pil)
            od_roi_tensor = F.to_tensor(od_roi_pil)
            
        return full_img_tensor, od_roi_tensor, label
