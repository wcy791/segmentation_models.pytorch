
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2
import json
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pywt

from wavelet_unet import WaveletUnet # Import our new model
from segmentation_models_pytorch.losses import DiceLoss

# --- 1. Configuration ---
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ENCODER = 'resnet34'
PRETRAINED_WEIGHTS = 'imagenet'

# --- Selectable Wavelet Types ---
WAVELET_OPTIONS = ['haar', 'db4', 'coif2', 'sym5']
WAVELET_TYPE = WAVELET_OPTIONS[0]  # Change index to select: 0:haar, 1:db4, 2:coif2, 3:sym5

print(f"--- Using Wavelet: {WAVELET_TYPE} ---")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train')
TRAIN_ANN_DIR = os.path.join(DATA_DIR, 'annotations_train')
VAL_IMG_DIR = os.path.join(DATA_DIR, 'val')
VAL_ANN_DIR = os.path.join(DATA_DIR, 'annotations_val')

CLASS_NAMES = ['background', 'black', 'blue', 'orange', 'pink']
CLASSES = {name: i for i, name in enumerate(CLASS_NAMES) if i > 0}

# --- 2. Dataset Definition with Wavelet Transform ---
class WaveletRespiratorDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, augmentations=None, image_size=(512, 512)):
        annot_basenames = {os.path.splitext(f)[0] for f in os.listdir(annotation_dir)}
        all_image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.image_paths = [
            os.path.join(image_dir, f) for f in all_image_files 
            if os.path.splitext(f)[0] in annot_basenames
        ]
        self.annotation_dir = annotation_dir
        self.augmentations = augmentations
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        basename = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(self.annotation_dir, basename + '.xml')
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # --- Create Mask ---
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        with open(json_path, 'r') as f:
            data = json.load(f)
        for shape in data['shapes']:
            if shape['label'] in CLASSES:
                points = np.array(shape['points'])
                cv2.fillPoly(mask, [points.astype(int)], color=CLASSES[shape['label']])

        # --- Apply Augmentations ---
        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # --- Wavelet Transform ---
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        coeffs = pywt.dwt2(gray_image, WAVELET_TYPE)
        LL, (LH, HL, HH) = coeffs
        
        # Stack detail coefficients. They are already float. Normalize to [-1, 1] range for stability.
        wavelet_features = np.stack([
            (LH - np.mean(LH)) / (np.std(LH) + 1e-6),
            (HL - np.mean(HL)) / (np.std(HL) + 1e-6),
            (HH - np.mean(HH)) / (np.std(HH) + 1e-6)
        ], axis=-1)

        # --- Preprocessing and Tensor Conversion ---
        # Convert image to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        wavelet_features = wavelet_features.astype(np.float32)

        # Transpose to [C, H, W] for PyTorch
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1))
        wavelet_tensor = torch.from_numpy(wavelet_features.transpose(2, 0, 1))
        mask_tensor = torch.from_numpy(mask).long()

        return image_tensor, wavelet_tensor, mask_tensor

# --- 3. Training and Evaluation Functions ---
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for image, wavelet, mask in tqdm(dataloader, desc="Training"):
        image, wavelet, mask = image.to(device), wavelet.to(device), mask.to(device)
        
        optimizer.zero_grad()
        outputs = model(image, wavelet)
        loss = loss_fn(outputs, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for image, wavelet, mask in tqdm(dataloader, desc="Evaluating"):
            image, wavelet, mask = image.to(device), wavelet.to(device), mask.to(device)
            outputs = model(image, wavelet)
            loss = loss_fn(outputs, mask)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# --- 4. Main Execution ---
if __name__ == '__main__':
    print(f"Using device: {DEVICE}")
    print("Training Wavelet-Unet Model")

    train_augs = A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
    ])
    val_augs = A.Compose([A.Resize(512, 512)])

    train_dataset = WaveletRespiratorDataset(TRAIN_IMG_DIR, TRAIN_ANN_DIR, augmentations=train_augs)
    val_dataset = WaveletRespiratorDataset(VAL_IMG_DIR, VAL_ANN_DIR, augmentations=val_augs)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = WaveletUnet(
        encoder_name=ENCODER, 
        encoder_weights=PRETRAINED_WEIGHTS, 
        classes=len(CLASS_NAMES)
    ).to(DEVICE)
    
    loss_fn = DiceLoss(mode='multiclass', from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        val_loss = evaluate(model, val_loader, loss_fn, DEVICE)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'wavelet_unet_model.pth')
            print("Model saved: wavelet_unet_model.pth")
            
    print(f"\n--- Training finished! Final model saved as 'wavelet_unet_model.pth' with best validation loss: {best_val_loss:.4f} ---")
