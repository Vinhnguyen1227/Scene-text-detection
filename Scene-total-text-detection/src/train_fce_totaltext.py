import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm
import warnings

# MMOCR Imports
from mmengine.config import Config
from mmocr.registry import MODELS
from mmocr.utils import register_all_modules

register_all_modules()

warnings.filterwarnings('ignore')

# Radial Encoder Target
def get_radial_encoded_target(polygon, cx, cy, num_rays=20):
    pts = np.array(polygon).reshape(-1, 2)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    distances = []
    for i in range(num_rays):
        angle = i * 2 * np.pi / num_rays
        max_d = 0.0
        vx, vy = np.cos(angle), np.sin(angle)

        for j in range(len(pts) - 1):
            x1, y1 = pts[j]
            x2, y2 = pts[j + 1]

            dx, dy = x2 - x1, y2 - y1

            det = vx * -dy - vy * -dx
            if abs(det) < 1e-6:
                continue

            d = (-dy * (x1 - cx) - -dx * (y1 - cy)) / det
            t = (-vy * (x1 - cx) + vx * (y1 - cy)) / det

            if d >= 0 and 0 <= t <= 1:
                if d > max_d:
                    max_d = d

        distances.append(max_d)

    return distances



# Helper to evaluate IoU for DetEval
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def decode_radial_to_box(cx, cy, dists):
    num_rays = len(dists)
    pts = []
    for i in range(num_rays):
        angle = i * 2 * np.pi / num_rays
        pts.append([cx + dists[i] * np.cos(angle), cy + dists[i] * np.sin(angle)])
    pts = np.array(pts)
    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)
    return [xmin, ymin, xmax, ymax]


# Custom Radial Head
class RadialHead(nn.Module):
    def __init__(self, in_channels=256, num_rays=20):
        super().__init__()
        self.num_rays = num_rays
        
        # P3, P4, P5 are the 3 outputs of the FPN in FCENet config
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_rays, 1)
        )

    def forward(self, feats):
        outputs = []
        for f in feats:
            outputs.append(self.conv(f))
        return outputs


# PyTorch Dataset for TotalText
class TotalTextDataset(Dataset):
    def __init__(self, img_dir, ann_dir, img_size=512, num_rays=20):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.img_size = img_size
        self.num_rays = num_rays
        
        self.image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        
        # ImageNet Normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.image_paths)
    
    def parse_txt_annotation(self, ann_path, orig_w, orig_h):
        polygons = []
        if not os.path.exists(ann_path):
            return polygons
            
        with open(ann_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        pattern = re.compile(r"x: \[\[(.*?)\]\],\s*y: \[\[(.*?)\]\]")
        for line in lines:
            line = line.strip()
            if not line: continue
            
            match = pattern.search(line)
            if match:
                x_str = match.group(1)
                y_str = match.group(2)
                
                x_vals = [float(v) for v in x_str.replace(',', ' ').split() if v.strip()]
                y_vals = [float(v) for v in y_str.replace(',', ' ').split() if v.strip()]
                
                if len(x_vals) > 0 and len(x_vals) == len(y_vals):
                    # Scale coordinates to the new image size (512x512)
                    x_vals = [x * (self.img_size / orig_w) for x in x_vals]
                    y_vals = [y * (self.img_size / orig_h) for y in y_vals]
                    
                    poly = np.column_stack((x_vals, y_vals))
                    polygons.append(poly)
                    
        return polygons

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        img_id = filename.split('.')[0] # e.g., img11
        
        # Load image (BGR to RGB)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
            
        orig_h, orig_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to square
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Normalize ImageNet
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1)) # (C, H, W)
        
        # Load annotations
        ann_path = os.path.join(self.ann_dir, f"poly_gt_{img_id}.txt")
        polygons = self.parse_txt_annotation(ann_path, orig_w, orig_h)
        
        # Generate Ground Truth distance maps
        gt_targets = {}
        strides = [8, 16, 32] # P3, P4, P5
        
        for stride in strides:
            f_size = self.img_size // stride
            
            # Mask tracking where text exists (1 for text, 0 for bg)
            mask = np.zeros((1, f_size, f_size), dtype=np.float32)
            # Distance vectors
            distances = np.zeros((self.num_rays, f_size, f_size), dtype=np.float32)
            
            for poly in polygons:
                poly_stride = poly / stride
                
                xmin, ymin = np.min(poly_stride, axis=0)
                xmax, ymax = np.max(poly_stride, axis=0)
                
                # Bounding box center
                cx = (xmin + xmax) / 2.0
                cy = (ymin + ymax) / 2.0
                
                # Get distance targets for this polygon at this stride
                dists = get_radial_encoded_target(poly_stride, cx, cy, self.num_rays)
                
                # Mark a small region around center as positive
                # In standard models, you compute distances for every point in the polygon
                # For simplicity in this script, we'll mark a 3x3 region at the true center
                icx, icy = int(np.clip(cx, 0, f_size-1)), int(np.clip(cy, 0, f_size-1))
                
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = icy + dy, icx + dx
                        if 0 <= ny < f_size and 0 <= nx < f_size:
                            mask[0, ny, nx] = 1.0
                            distances[:, ny, nx] = dists
                            
            gt_targets[f"stride_{stride}"] = {
                "mask": torch.from_numpy(mask),
                "dist": torch.from_numpy(distances)
            }
            
        return torch.from_numpy(img), gt_targets


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    # Restructure targets to batched tensors
    batched_targets = {}
    for stride in [8, 16, 32]:
        masks = torch.stack([t[f"stride_{stride}"]["mask"] for t in targets], dim=0)
        dists = torch.stack([t[f"stride_{stride}"]["dist"] for t in targets], dim=0)
        batched_targets[f"stride_{stride}"] = {"mask": masks, "dist": dists}
    return images, batched_targets


# Define Dynamic Building and Training
def train():
    import os
    # Fix PyTorch Memory fragmentation
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths
    config_file = r"D:\computer vision\Scene-total-text-detection\src\fcenet_resnet50-dcnv2_fpn_1500e_ctw1500.py"
    checkpoint = r"D:\computer vision\Scene-total-text-detection\fcenet_resnet50-dcnv2_fpn_1500e_ctw1500_20220825_221510-4d705392.pth" 

    train_img_dir = r"D:\computer vision\Scene-total-text-detection\dataset\totaltext\train\Images"
    train_ann_dir = r"D:\computer vision\Scene-total-text-detection\dataset\totaltext\train\annotations"
    val_img_dir = r"D:\computer vision\Scene-total-text-detection\dataset\totaltext\test\Images"
    val_ann_dir = r"D:\computer vision\Scene-total-text-detection\dataset\totaltext\test\annotations"
    
    num_rays = 20
    img_size = 512
    batch_size = 8
    epochs = 100

    # 1. Build Model from Config
    print("Loading MMOCR Config...")
    cfg = Config.fromfile(config_file)
    model = MODELS.build(cfg.model)

    # 2. Replace FCE Head with our custom RadialHead
    print("Injecting Radial Head...")
    model.det_head = RadialHead(in_channels=256, num_rays=num_rays)

    # 3. Load Backbone/Neck Weights
    print("Loading Pretrained Checkpoint...")
    ckpt = torch.load(checkpoint, map_location="cpu")
    state_dict = ckpt["state_dict"]

    filtered = {
        k: v for k, v in state_dict.items()
        if not k.startswith("det_head")
    }
    model.load_state_dict(filtered, strict=False)

    # Freeze backbone and neck
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.neck.parameters():
        param.requires_grad = False

    model = model.to(device)

    # 4. Prepare DataLoader
    print("Initializing Dataset...")
    train_dataset = TotalTextDataset(img_dir=train_img_dir, ann_dir=train_ann_dir, 
                                     img_size=img_size, num_rays=num_rays)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)

    print("Initializing Validation Dataset...")
    val_dataset = TotalTextDataset(img_dir=val_img_dir, ann_dir=val_ann_dir, 
                                   img_size=img_size, num_rays=num_rays)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True, collate_fn=collate_fn)

    # 5. Optimizer and Loss Function
    optimizer = torch.optim.AdamW(model.det_head.parameters(), lr=1e-4)
    loss_fn = nn.SmoothL1Loss(reduction='none') # Compute loss per pixel, then mask

    # 6. Training Loop
    print(f"Starting Training for {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, targets in pbar:
            images = images.to(device).float()
            masks_8 = targets["stride_8"]["mask"].to(device)
            dists_8 = targets["stride_8"]["dist"].to(device)
            
            masks_16 = targets["stride_16"]["mask"].to(device)
            dists_16 = targets["stride_16"]["dist"].to(device)
            
            masks_32 = targets["stride_32"]["mask"].to(device)
            dists_32 = targets["stride_32"]["dist"].to(device)

            optimizer.zero_grad()

            # Forward pass through MMOCR network (wrapping in autocast for DCNv2 compat)
            with torch.no_grad():
                with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    feats_backbone = model.backbone(images)
                    feats_neck = model.neck(feats_backbone)
                
            # Forward pass through our head
            with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                preds = model.det_head(feats_neck) 
            
            # Predicts list of [P3, P4, P5] features which correspond to Strides 8, 16, 32
            pred_8, pred_16, pred_32 = preds[0], preds[1], preds[2]
            
            # Compute loss only on masked positive center locations
            loss_8 = (loss_fn(pred_8, dists_8) * masks_8).sum() / (masks_8.sum() + 1e-6)
            loss_16 = (loss_fn(pred_16, dists_16) * masks_16).sum() / (masks_16.sum() + 1e-6)
            loss_32 = (loss_fn(pred_32, dists_32) * masks_32).sum() / (masks_32.sum() + 1e-6)
            
            total_loss = loss_8 + loss_16 + loss_32
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            pbar.set_postfix({"Loss": f"{total_loss.item():.4f}"})
            
        # Validation / DetEval Loop
        model.eval()
        val_loss = 0.0
        total_gt_boxes = 0
        correct_preds = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device).float()
                masks_8 = targets["stride_8"]["mask"].to(device)
                dists_8 = targets["stride_8"]["dist"].to(device)
                
                with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    feats_backbone = model.backbone(images)
                    feats_neck = model.neck(feats_backbone)
                    preds = model.det_head(feats_neck)
                
                pred_8 = preds[0] # stride 8
                val_loss += ((loss_fn(pred_8, dists_8) * masks_8).sum() / (masks_8.sum() + 1e-6)).item()
                
                # DetEval calculation (pseudo-NMS by taking masked centers)
                b = pred_8.shape[0]
                for i in range(b):
                    gt_ys, gt_xs = torch.where(masks_8[i, 0] > 0.5)
                    gt_boxes = []
                    pred_boxes = []
                    for gy, gx in zip(gt_ys, gt_xs):
                        gt_dists = dists_8[i, :, gy, gx].cpu().numpy()
                        gt_boxes.append(decode_radial_to_box(gx.item(), gy.item(), gt_dists))
                        
                        p_dists = pred_8[i, :, gy, gx].cpu().numpy()
                        pred_boxes.append(decode_radial_to_box(gx.item(), gy.item(), p_dists))
                        
                    total_gt_boxes += len(gt_boxes)
                    
                    # Match predictions to GT with IoU > 0.5
                    matched_gt = set()
                    for pb in pred_boxes:
                        best_iou = 0
                        best_gt_idx = -1
                        for idx, gb in enumerate(gt_boxes):
                            if idx in matched_gt: continue
                            iou = compute_iou(pb, gb)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = idx
                        if best_iou > 0.5:
                            correct_preds += 1
                            matched_gt.add(best_gt_idx)
                            
        precision = correct_preds / max(total_gt_boxes, 1) # Simplified for eval
        recall = precision # Since we only predict strictly against GT centers here
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
                        
        print(f"\nEpoch [{epoch+1}/{epochs}] - Train Loss: {epoch_loss / len(train_loader):.4f} | Val Loss: {val_loss / len(val_loader):.4f}")
        print(f"DetEval - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")
        
        model.train()
        
    print("Training Complete. Saving model...")
    torch.save(model.state_dict(), "fcenet_radial_head_finetuned.pth")


if __name__ == '__main__':
    train()