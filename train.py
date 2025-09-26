import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2
import json
from tqdm import tqdm

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

# --- 1. 配置 ---
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS_SUPERVISED = 15
EPOCHS_SEMI_SUPERVISED = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ENCODER = 'resnet34'
PRETRAINED_WEIGHTS = 'imagenet'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train')
TRAIN_ANN_DIR = os.path.join(DATA_DIR, 'annotations_train')
VAL_IMG_DIR = os.path.join(DATA_DIR, 'val')
VAL_ANN_DIR = os.path.join(DATA_DIR, 'annotations_val')

CLASS_NAMES = ['background', 'black', 'blue', 'orange', 'pink']
CLASSES = {name: i for i, name in enumerate(CLASS_NAMES) if i > 0}
IGNORE_INDEX = 255
PSEUDO_LABEL_CONFIDENCE_THRESHOLD = 0.9

# --- 2. 数据集定义 ---

class RespiratorDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, image_size=(512, 512), preprocessing_fn=None):
        # Correctly filter for images that have a corresponding annotation file
        annot_basenames = {os.path.splitext(f)[0] for f in os.listdir(annotation_dir)}
        all_image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.image_paths = [
            os.path.join(image_dir, f) for f in all_image_files 
            if os.path.splitext(f)[0] in annot_basenames
        ]
        
        self.annotation_dir = annotation_dir
        self.image_size = image_size
        self.preprocessing_fn = preprocessing_fn

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        basename = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(self.annotation_dir, basename + '.xml')
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        
        mask = np.zeros(self.image_size, dtype=np.uint8)
        # No need to check os.path.exists(json_path) because the constructor guarantees it
        with open(json_path, 'r') as f:
            data = json.load(f)
        for shape in data['shapes']:
            if shape['label'] in CLASSES:
                points = np.array(shape['points'])
                points[:, 0] *= (self.image_size[0] / data['imageWidth'])
                points[:, 1] *= (self.image_size[1] / data['imageHeight'])
                cv2.fillPoly(mask, [points.astype(int)], color=CLASSES[shape['label']])
        
        if self.preprocessing_fn:
            image = self.preprocessing_fn(image)
            
        image = image.transpose(2, 0, 1).astype('float32')
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).long()
        return image, mask

class SemiSupervisedDataset(Dataset):
    def __init__(self, train_img_dir, train_ann_dir, image_size=(512, 512), preprocessing_fn=None, pseudo_label_model=None):
        self.labeled_dataset = RespiratorDataset(train_img_dir, train_ann_dir, image_size, preprocessing_fn)
        
        annot_basenames = {os.path.splitext(f)[0] for f in os.listdir(train_ann_dir)}
        all_train_images = [os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir) if f.endswith('.jpg')]
        self.unlabeled_paths = [p for p in all_train_images if os.path.splitext(os.path.basename(p))[0] not in annot_basenames]
        
        self.image_size = image_size
        self.preprocessing_fn = preprocessing_fn
        self.pseudo_label_model = pseudo_label_model
        if self.pseudo_label_model:
            self.pseudo_label_model.eval()

    def __len__(self):
        return len(self.labeled_dataset) + len(self.unlabeled_paths)

    def __getitem__(self, idx):
        if idx < len(self.labeled_dataset):
            return self.labeled_dataset[idx]
        else:
            image_path = self.unlabeled_paths[idx - len(self.labeled_dataset)]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.image_size)
            
            processed_image = self.preprocessing_fn(image.copy()) if self.preprocessing_fn else image.copy()
            input_tensor = torch.from_numpy(processed_image.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
            
            with torch.no_grad():
                logits = self.pseudo_label_model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                max_probs, pred_classes = torch.max(probs, dim=1)
                pred_classes[max_probs < PSEUDO_LABEL_CONFIDENCE_THRESHOLD] = IGNORE_INDEX
                mask = pred_classes.squeeze(0).cpu().long()

            return torch.from_numpy(processed_image.transpose(2,0,1)).float(), mask

# --- 3. 训练和评估辅助函数 ---
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.long)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.long)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# --- 4. 主执行流程 ---
if __name__ == '__main__':
    print(f"使用设备: {DEVICE}")
    print("执行方案: 半监督学习 (预训练U-Net) + 独立验证集")

    smp_preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, pretrained=PRETRAINED_WEIGHTS)

    # --- 阶段 1: 监督学习 ---
    print("\n--- 开始阶段 1: 监督学习 ---")
    supervised_model = smp.Unet(encoder_name=ENCODER, encoder_weights=PRETRAINED_WEIGHTS, in_channels=3, classes=len(CLASS_NAMES)).to(DEVICE)
    loss_fn = DiceLoss(mode='multiclass', from_logits=True)
    optimizer = torch.optim.Adam(supervised_model.parameters(), lr=LEARNING_RATE)
    
    train_dataset = RespiratorDataset(TRAIN_IMG_DIR, TRAIN_ANN_DIR, preprocessing_fn=smp_preprocessing_fn)
    val_dataset = RespiratorDataset(VAL_IMG_DIR, VAL_ANN_DIR, preprocessing_fn=smp_preprocessing_fn)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"阶段1: 训练集样本数: {len(train_dataset)}")
    print(f"阶段1: 验证集样本数: {len(val_dataset)}")

    best_val_loss = float('inf')
    for epoch in range(EPOCHS_SUPERVISED):
        print(f"\nEpoch {epoch + 1}/{EPOCHS_SUPERVISED}")
        train_loss = train_one_epoch(supervised_model, train_loader, optimizer, loss_fn, DEVICE)
        val_loss = evaluate(supervised_model, val_loader, loss_fn, DEVICE)
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(supervised_model.state_dict(), 'initial_model.pth')
            print("模型已保存: initial_model.pth")

    print("\n--- 阶段 1 完成 ---")

    # --- 阶段 2: 半监督学习 ---
    print("\n--- 开始阶段 2: 半监督学习 ---")
    pseudo_label_model = smp.Unet(encoder_name=ENCODER, encoder_weights=None, in_channels=3, classes=len(CLASS_NAMES)).to(DEVICE)
    pseudo_label_model.load_state_dict(torch.load('initial_model.pth', weights_only=True))
    
    semi_supervised_model = smp.Unet(encoder_name=ENCODER, encoder_weights=PRETRAINED_WEIGHTS, in_channels=3, classes=len(CLASS_NAMES)).to(DEVICE)
    semi_supervised_model.load_state_dict(torch.load('initial_model.pth', weights_only=True))
    
    loss_fn_semi = DiceLoss(mode='multiclass', from_logits=True, ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(semi_supervised_model.parameters(), lr=LEARNING_RATE / 2)

    semi_dataset = SemiSupervisedDataset(TRAIN_IMG_DIR, TRAIN_ANN_DIR, preprocessing_fn=smp_preprocessing_fn, pseudo_label_model=pseudo_label_model)
    semi_train_loader = DataLoader(semi_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    print(f"阶段2: 训练集样本数 (有标签+无标签): {len(semi_dataset)}")
    print(f"阶段2: 验证集样本数: {len(val_dataset)}")

    best_val_loss = float('inf')
    for epoch in range(EPOCHS_SEMI_SUPERVISED):
        print(f"\nEpoch {epoch + 1}/{EPOCHS_SEMI_SUPERVISED}")
        train_loss = train_one_epoch(semi_supervised_model, semi_train_loader, optimizer, loss_fn_semi, DEVICE)
        val_loss = evaluate(semi_supervised_model, val_loader, loss_fn, DEVICE)
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(semi_supervised_model.state_dict(), 'final_semisupervised_model.pth')
            print("模型已保存: final_semisupervised_model.pth")
            
    print(f"\n--- 训练完成！最终半监督模型已保存为 'final_semisupervised_model.pth' ---")
