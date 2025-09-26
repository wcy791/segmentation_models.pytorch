

import torch
import numpy as np
import cv2
import os
import glob
import sys
import pywt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from wavelet_unet import WaveletUnet # Import our custom model

# --- 1. Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'wavelet_unet_model.pth') 
TEST_IMAGE_DIR = os.path.join(SCRIPT_DIR, 'data', 'test')

ENCODER = 'resnet34'
PRETRAINED_WEIGHTS = 'imagenet'
WAVELET_TYPE = 'haar'
CLASS_NAMES = ['background', 'black', 'blue', 'orange', 'pink']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. Load WaveletUnet Model ---
print(f"Using device: {DEVICE}")
if not os.path.exists(MODEL_PATH):
    print(f"Error: Wavelet model file not found at '{MODEL_PATH}'")
    print("Please run train_wavelet_unet.py script first.")
    exit()

model = WaveletUnet(
    encoder_name=ENCODER,
    encoder_weights=PRETRAINED_WEIGHTS,
    classes=len(CLASS_NAMES)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()
print("WaveletUnet model loaded successfully.")

# --- 3. Load and Preprocess Image ---
IMAGE_PATH = None
if len(sys.argv) > 1:
    IMAGE_PATH = sys.argv[1]
else:
    test_images = glob.glob(os.path.join(TEST_IMAGE_DIR, '*.jpg'))
    if test_images:
        IMAGE_PATH = test_images[0]

if not IMAGE_PATH or not os.path.exists(IMAGE_PATH):
    print(f"Error: Test image '{IMAGE_PATH}' not found or path is invalid.")
    exit()

print(f"Evaluating image: {IMAGE_PATH}")

original_image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# --- Preprocessing consistent with training ---
# 1. Resize
resize_aug = A.Compose([A.Resize(512, 512)])
resized_image = resize_aug(image=image_rgb)['image']

# 2. Create Wavelet Features
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
coeffs = pywt.dwt2(gray_image, WAVELET_TYPE)
LL, (LH, HL, HH) = coeffs
wavelet_features = np.stack([
    (LH - np.mean(LH)) / (np.std(LH) + 1e-6),
    (HL - np.mean(HL)) / (np.std(HL) + 1e-6),
    (HH - np.mean(HH)) / (np.std(HH) + 1e-6)
], axis=-1)

# 3. Convert to Tensors
image = resized_image.astype(np.float32) / 255.0
wavelet_features = wavelet_features.astype(np.float32)

image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
wavelet_tensor = torch.from_numpy(wavelet_features.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

# --- 4. Perform Inference ---
print("\nPerforming model inference...")
with torch.no_grad():
    output = model(image_tensor, wavelet_tensor)
pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

# --- 5. Analysis ---
print("\n--- Model Evaluation Result ---")
pixel_counts = {}
for i, name in enumerate(CLASS_NAMES):
    if i == 0: continue
    count = np.sum(pred_mask == i)
    if count > 10:
        pixel_counts[name] = count

if not pixel_counts:
    print("模型未在此图片中检测到任何有效颜色区域。")
else:
    print("模型在图片中检测到以下颜色及其占比:")
    total_detected_pixels = sum(pixel_counts.values())
    # 按照像素数降序排列，方便查看主要颜色
    sorted_colors = sorted(pixel_counts.items(), key=lambda item: item[1], reverse=True)
    
    for name, count in sorted_colors:
        percentage = (count / total_detected_pixels) * 100
        print(f"- {name.capitalize():<8}: {count:>8} 像素, (占已检测颜色的 {percentage:.2f}%)")

# --- 6. Visualization ---
# Using the corrected BGR color map for OpenCV
color_map = np.array([
    [0, 0, 0],          # background -> black
    [128, 128, 128],    # black -> gray
    [255, 0, 0],        # blue -> blue
    [0, 165, 255],      # orange -> orange
    [203, 192, 255]     # pink -> pink
], dtype=np.uint8)

color_mask = color_map[pred_mask.astype(np.uint8)]

# Resize original image to match the mask size for side-by-side comparison
resized_original_for_overlay = cv2.resize(original_image, (pred_mask.shape[1], pred_mask.shape[0]))

comparison_image = np.hstack([resized_original_for_overlay, color_mask])

base_name = os.path.basename(IMAGE_PATH)
file_name_without_ext = os.path.splitext(base_name)[0]
output_filename = f"evaluation_wavelet_{file_name_without_ext}.png"

cv2.imwrite(output_filename, comparison_image)

print(f"\nEvaluation visualization saved to '{output_filename}'")
print("Left: Original Image, Right: Predicted Segmentation Mask")
