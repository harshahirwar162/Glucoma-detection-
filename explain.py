import cv2
import numpy as np
import torch
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from .model import MultiBackboneFundusModel
from .preprocessing import preprocess_pipeline

class DualInputTarget:
    """
    Wrapper required by pytorch-grad-cam library to handle models 
    that take multiple inputs during the forward pass.
    """
    def __init__(self, model):
        self.model = model
        
    def __call__(self, x_full, x_roi):
        return self.model(x_full, x_roi)

def visualize_gradcam(image_path: str, model_weights_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = MultiBackboneFundusModel(num_classes=5)
    if model_weights_path:
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Prepare Inputs
    original_bgr = cv2.imread(image_path)
    if original_bgr is None:
        raise ValueError("Image not found.")
    rgb_img = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    full_img, od_roi = preprocess_pipeline(rgb_img)
    
    # Normalize for Visualization (GradCAM show_cam_on_image needs float32 inputs between 0-1)
    full_img_vis = cv2.resize(full_img, (224, 224)) / 255.0
    od_roi_vis = cv2.resize(od_roi, (224, 224)) / 255.0
    
    # Transfoms for model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor_full = transform(full_img).unsqueeze(0).to(device)
    input_tensor_roi = transform(od_roi).unsqueeze(0).to(device)
    
    # We must wrap the model forward pass slightly because pytorch_grad_cam natively
    # supports single-input models. However, we can use specific target layers.
    
    # Define target layers for both backbones
    # EfficientNet target: last conv block
    target_layer_effnet = [model.branch1.features[-1]]
    
    # ResNet50 target: last bottleneck block
    target_layer_resnet = [model.branch2.layer4[-1]]
    
    # --- Generate CAM for EfficientNet (Full Image Branch) ---
    print("Generating Grad-CAM for EfficientNet (Full Image) branch...")
    cam_effnet = GradCAM(model=model, target_layers=target_layer_effnet)
    
    # Custom forward pass closure inside GradCAM requires tuple if multiple inputs
    # Let's target class 0 for visualization (or replace with max prediction)
    targets = [ClassifierOutputTarget(1)] 
    
    # Run CAM
    grayscale_cam_effnet = cam_effnet(input_tensor=input_tensor_full, 
                                      targets=targets,
                                      eigen_smooth=False,
                                      aug_smooth=False)
    grayscale_cam_effnet = grayscale_cam_effnet[0, :]
    effnet_visualization = show_cam_on_image(full_img_vis, grayscale_cam_effnet, use_rgb=True)
    
    cv2.imwrite("gradcam_efficientnet_full.jpg", cv2.cvtColor(effnet_visualization, cv2.COLOR_RGB2BGR))
    print("Saved gradcam_efficientnet_full.jpg")

    # Note: GradCAM for ResNet branch would identically wrap target_layer_resnet and input_tensor_roi.
    
if __name__ == "__main__":
    # example usage
    # visualize_gradcam('test_image.png')
    pass
