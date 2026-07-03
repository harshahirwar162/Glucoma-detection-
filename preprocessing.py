import cv2
import numpy as np

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the image.
    Works on RGB images by converting to LAB, applying CLAHE to the L channel, and converting back.
    """
    if len(image.shape) == 2:  # Grayscale
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
    
    # Color image
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    converted = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return converted

def extract_optic_disc_roi(image: np.ndarray, roi_size: tuple = (224, 224)) -> np.ndarray:
    """
    Approximates Optic Disc location based on thresholding the Red channel (typically brightest).
    Crops an ROI of size `roi_size` around the detected center.
    """
    # Using Red channel to find the optic disc (usually highest intensity)
    if len(image.shape) == 3:
        r_channel = image[:, :, 0]
    else:
        r_channel = image
        
    # Gaussian Blur to smooth structures
    blurred = cv2.GaussianBlur(r_channel, (15, 15), 0)
    
    # Thresholding to find the brightest spots (Optic Disc)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback to center of image if no bright spot found
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
    else:
        # Get largest contour assuming it's the OD
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            h, w = image.shape[:2]
            cx, cy = w // 2, h // 2

    # Crop parameters
    rw, rh = roi_size
    h, w = image.shape[:2]
    
    x_start = max(0, cx - rw // 2)
    y_start = max(0, cy - rh // 2)
    x_end = min(w, x_start + rw)
    y_end = min(h, y_start + rh)
    
    # Adjust if near borders
    if (x_end - x_start) < rw:
        x_start = max(0, x_end - rw)
    if (y_end - y_start) < rh:
        y_start = max(0, y_end - rh)
        
    roi = image[y_start:y_end, x_start:x_end]
    
    # Resize just in case it's still smaller than roi_size due to small image size
    if roi.shape[:2] != (rh, rw):
        roi = cv2.resize(roi, (rw, rh))
        
    return roi

def apply_mca(image: np.ndarray) -> np.ndarray:
    """
    Morphological Component Analysis (MCA) splits an image into morphological parts 
    (e.g., texture vs structure).
    
    NOTE: This is a placeholder as true MCA requires a complex iterative thresholding 
    algorithm over dictionaries (like Curvelet/Wavelet transforms).
    We apply a high-pass / low-pass filter logic as a very basic approximation.
    """
    # Approximation: Returning high-frequency component
    blurred = cv2.GaussianBlur(image, (21, 21), 0)
    structure_component = blurred
    texture_component = cv2.subtract(image, blurred)
    
    # The actual implementation depends on mathematical toolkits, normally handled
    # via separate libraries or custom PyWavelets iterative solvers.
    return texture_component

def apply_2d_fbse_ewt(image: np.ndarray) -> np.ndarray:
    """
    2D Fourier-Bessel Series Expansion Empirical Wavelet Transform.
    This is highly specialized mathematical feature enhancement.
    
    NOTE: Placeholder. Returns original image. Real implementation requires 
    Bessel functions and custom filter banks in frequency domain.
    """
    # TODO: Integrate official FBSE mathematical code if available.
    return image

def preprocess_pipeline(image: np.ndarray) -> tuple:
    """
    Executes the full preprocessing pipeline on a single image.
    Returns:
        full_processed_image, optic_disc_roi
    """
    # 1. CLAHE
    img_clahe = apply_clahe(image)
    
    # 2. Extract OD ROI from CLAHE image
    od_roi = extract_optic_disc_roi(img_clahe)
    
    # 3. MCA & FBSE EWT (Applied to full image for structure/texture)
    img_mca = apply_mca(img_clahe)
    img_final = apply_2d_fbse_ewt(img_mca)
    
    return img_final, od_roi
