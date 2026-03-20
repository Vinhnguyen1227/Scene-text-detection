import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import csv
import glob
import numpy as np
import warnings
from mmocr.apis import MMOCRInferencer

warnings.filterwarnings('ignore')

def main():
    # Paths configuration
    config_file = r"D:\computer vision\Scene-total-text-detection\src\fcenet_resnet50-dcnv2_fpn_1500e_ctw1500.py"
    checkpoint = r"D:\computer vision\Scene-total-text-detection\fcenet_resnet50-dcnv2_fpn_1500e_ctw1500_20220825_221510-4d705392.pth"
    test_dir = r"D:\computer vision\Scene-total-text-detection\test"
    output_dir = r"D:\computer vision\Scene-total-text-detection\output"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "predictions.csv")

    # Image Paths
    image_paths = sorted(glob.glob(os.path.join(test_dir, "*.jpg")) + glob.glob(os.path.join(test_dir, "*.png")))
    if not image_paths:
        print(f"No images found in {test_dir}!")
        return

    # Initialize MMOCR Inferencer
    print("Loading FCENet from provided weights...")
    inferencer = MMOCRInferencer(det=config_file, det_weights=checkpoint, device='cuda')

    # Prepare CSV header for exactly 14 vertices (28 coordinates)
    csv_header = ['image_name']
    for i in range(1, 15):
        csv_header.extend([f'x{i}', f'y{i}'])
    
    print(f"Starting inference on {len(image_paths)} images...")
    
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            print(f"Processing {img_name}...")
            
            # Read image for drawing later
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping {img_name}: Unable to read image")
                continue
                
            # Run Inference
            # The result is a dictionary containing "predictions" (with "det_polygons" and "det_scores")
            result = inferencer(img_path, return_vis=False)
            
            det_polygons = result['predictions'][0]['det_polygons']
            det_scores = result['predictions'][0]['det_scores']
            
            for poly, score in zip(det_polygons, det_scores):
                # Filter low-confidence detections
                if score < 0.5:
                    continue
                
                # Reshape to expected format: [[x1, y1], [x2, y2], ...]
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                
                # Ensure we strictly have 14 vertices by resampling if needed
                if len(pts) != 14:
                    num_pts = len(pts)
                    indices = np.linspace(0, num_pts - 1, 14).astype(int)
                    resampled_pts = pts[indices]
                else:
                    resampled_pts = pts
                
                # Flatten the 14 vertices for CSV row: [x1, y1, x2, y2... x14, y14]
                row_data = [img_name] + resampled_pts.flatten().tolist()
                writer.writerow(row_data)
                
                # Draw the polygon on the image
                cv2.polylines(img, [resampled_pts], isClosed=True, color=(0, 255, 0), thickness=2)
                
                # Draw vertices as points
                for pt in resampled_pts:
                    cv2.circle(img, tuple(pt), radius=3, color=(0, 0, 255), thickness=-1)
            
            # Save Visualized Image
            out_img_path = os.path.join(output_dir, img_name)
            cv2.imwrite(out_img_path, img)

    print(f"\nInference Complete! Results saved to:")
    print(f"- CSV file: {csv_path}")
    print(f"- Visualizations: {output_dir}")


if __name__ == '__main__':
    main()
