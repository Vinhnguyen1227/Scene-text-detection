import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import torch
import warnings
from mmengine.config import Config
from mmocr.registry import MODELS
from mmocr.utils import register_all_modules
import torch.nn as nn

register_all_modules()
warnings.filterwarnings('ignore')

class RadialHead(nn.Module):
    def __init__(self, in_channels=256, num_rays=20):
        super().__init__()
        self.num_rays = num_rays
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware allocated: {device}")

    # Base configuration mapping
    config_file = r"D:\computer vision\Scene-total-text-detection\src\fcenet_resnet50-dcnv2_fpn_1500e_ctw1500.py"
    
    # 1. Build Original Backbone + Neck
    print("Mounting graph architecture...")
    cfg = Config.fromfile(config_file)
    model = MODELS.build(cfg.model)

    # 2. Swap FCE Head for Custom Radial Head
    model.det_head = RadialHead(in_channels=256, num_rays=20)
    model = model.to(device)
    model.eval()

    # Image geometry matching training dimensions
    img_size = 512
    batch_size = 1 # FPS is evaluated typically at Batch Size 1 for real-time cameras
    
    print(f"Generating Random Image Simulation Batch [(B, C, H, W) = ({batch_size}, 3, {img_size}, {img_size})]")
    dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(device)

    # Warm-up phase (prevents GPU initialization latency spikes)
    print("Running GPU warmup (50 iterations)...")
    with torch.no_grad():
        for _ in range(50):
            with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                feats_backbone = model.backbone(dummy_input)
                feats_neck = model.neck(feats_backbone)
                preds = model.det_head(feats_neck)
                
    # Benchmarking phase
    num_iterations = 500
    print(f"Initiating High-Speed Benchmark Queue ({num_iterations} sequences)...")

    # Accurate Timing using CUDA Event Synchronizations
    if torch.cuda.is_available():
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
    else:
        start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_iterations):
            with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                feats_backbone = model.backbone(dummy_input)
                feats_neck = model.neck(feats_backbone)
                preds = model.det_head(feats_neck)

    if torch.cuda.is_available():
        ender.record()
        torch.cuda.synchronize()
        total_time = starter.elapsed_time(ender) / 1000.0 # Convert ms -> seconds
    else:
        total_time = time.perf_counter() - start_time

    # Calculate metrics
    avg_latency_ms = (total_time / num_iterations) * 1000
    fps = 1.0 / (total_time / num_iterations)

    print("\n" + "="*50)
    print(" INFERENCE SPEED BENCHMARK REPORT")
    print("="*50)
    print(f" Backbone     : ResNet50 (DCNv2) FPN")
    print(f" Det Head     : Radial Regressor (20 Rays)")
    print(f" Input Shape  : {img_size}x{img_size} px")
    print("-" * 50)
    print(f" Total Time   : {total_time:.3f} seconds (for {num_iterations} images)")
    print(f" FPS Rate     : {fps:.2f} Frames per Second")
    print(f" Latency      : {avg_latency_ms:.2f} ms per Image")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
