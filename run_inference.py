import torch
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import os
import glob
import sys

# --- 1. 配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'final_semisupervised_model.pth') 
TEST_IMAGE_DIR = os.path.join(SCRIPT_DIR, 'data', 'test')

ENCODER = 'resnet34'
CLASS_NAMES = ['background', 'black', 'blue', 'orange', 'pink']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. 加载 smp.Unet 模型 ---
print(f"使用设备: {DEVICE}")
if not os.path.exists(MODEL_PATH):
    print(f"错误: 原始模型文件未找到！路径: '{MODEL_PATH}'")
    print("请先成功运行 train.py 脚本。")
    exit()

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=None,
    in_channels=3,
    classes=len(CLASS_NAMES),
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()
print("原始 smp.Unet 模型加载成功。")

# --- 3. 加载和预处理图像 ---
IMAGE_PATH = None
if len(sys.argv) > 1:
    IMAGE_PATH = sys.argv[1]
    print(f"从命令行指定测试图片: {IMAGE_PATH}")
else:
    test_images = glob.glob(os.path.join(TEST_IMAGE_DIR, '*.jpg'))
    if test_images:
        IMAGE_PATH = test_images[0]
        print(f"自动选择测试图片: {IMAGE_PATH}")

if not IMAGE_PATH or not os.path.exists(IMAGE_PATH):
    print(f"错误: 测试图片未找到或路径无效。路径: '{IMAGE_PATH}'")
    exit()

image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_rgb = cv2.resize(image_rgb, (512, 512))

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, "imagenet")
preprocessed_image = preprocessing_fn(image_rgb)
preprocessed_image = preprocessed_image.transpose(2, 0, 1).astype('float32')
image_tensor = torch.from_numpy(preprocessed_image).unsqueeze(0).to(DEVICE)

# --- 4. 执行推理 ---
print("\n正在执行模型推理...")
with torch.no_grad():
    output = model(image_tensor)
pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

# --- 5. 纯粹的评估和分析 ---
print("\n--- 模型评估结果 ---")
pixel_counts = {}
for i, name in enumerate(CLASS_NAMES):
    if i == 0: continue # 跳过背景
    count = np.sum(pred_mask == i)
    if count > 10: # 过滤掉少量噪声点
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

# --- 6. 可视化 ---
color_map = np.array([
    [0, 0, 0],          # background -> black
    [128, 128, 128],    # black -> gray
    [255, 0, 0],        # blue -> blue
    [0, 165, 255],      # orange -> orange
    [203, 192, 255]     # pink -> pink
], dtype=np.uint8)

color_mask = color_map[pred_mask]
resized_original_image = cv2.resize(image, (pred_mask.shape[1], pred_mask.shape[0]))

# 将分割结果和原图并排显示
comparison_image = np.hstack([resized_original_image, color_mask])

base_name = os.path.basename(IMAGE_PATH)
file_name_without_ext = os.path.splitext(base_name)[0]
output_filename = f"evaluation_original_{file_name_without_ext}.png"

cv2.imwrite(output_filename, comparison_image)

print(f"\n评估结果可视化已保存到 '{output_filename}'")
print("左边是原图，右边是模型预测的分割掩码。")
