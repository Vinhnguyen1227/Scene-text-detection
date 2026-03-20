import os
import glob
import re
import numpy as np
import cv2
import math

# Parse TotalText
def parse_txt_annotation(ann_path):
    polygons = []
    if not os.path.exists(ann_path): return polygons
    with open(ann_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    pattern = re.compile(r"x: \[\[(.*?)\]\],\s*y: \[\[(.*?)\]\]")
    for line in lines:
        line = line.strip()
        if not line: continue
        match = pattern.search(line)
        if match:
            x_vals = [float(v) for v in match.group(1).replace(',', ' ').split() if v.strip()]
            y_vals = [float(v) for v in match.group(2).replace(',', ' ').split() if v.strip()]
            if len(x_vals) > 0 and len(x_vals) == len(y_vals):
                poly = np.column_stack((x_vals, y_vals))
                polygons.append(poly)
    return polygons

# Original Radial Encoding
def get_radial_encoded_target(polygon, cx, cy, num_rays=20):
    pts = np.array(polygon).reshape(-1, 2)
    distances = []
    points = []
    for i in range(num_rays):
        angle = i * 2 * np.pi / num_rays
        max_d = 0.0
        vx, vy = np.cos(angle), np.sin(angle)
        best_pt = (cx, cy)
        for j in range(len(pts)):
            x1, y1 = pts[j]
            x2, y2 = pts[(j + 1) % len(pts)]
            dx, dy = x2 - x1, y2 - y1
            det = vx * -dy - vy * -dx
            if abs(det) < 1e-6: continue
            d = (-dy * (x1 - cx) - -dx * (y1 - cy)) / det
            t = (-vy * (x1 - cx) + vx * (y1 - cy)) / det
            if d >= 0 and 0 <= t <= 1:
                if d > max_d:
                    max_d = d
                    best_pt = (cx + d * vx, cy + d * vy)
        points.append(best_pt)
    return points

def visualize_radial(img, polygons, out_path):
    canvas = img.copy()
    for poly in polygons:
        poly_pts = np.array(poly, dtype=np.int32)
        cv2.polylines(canvas, [poly_pts], isClosed=True, color=(100, 255, 100), thickness=2)
        
        # Center point
        xmin, ymin = np.min(poly_pts, axis=0)
        xmax, ymax = np.max(poly_pts, axis=0)
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        
        # Draw Rays
        ray_pts = get_radial_encoded_target(poly, cx, cy, 20)
        for pt in ray_pts:
            cv2.arrowedLine(canvas, (int(cx), int(cy)), (int(pt[0]), int(pt[1])), (0, 165, 255), 2, tipLength=0.1)
        
        # Draw Center
        cv2.circle(canvas, (int(cx), int(cy)), radius=5, color=(0, 0, 255), thickness=-1)
        
    cv2.imwrite(out_path, canvas)
    print(f"Saved Radial Ray visualization to {out_path}")


def visualize_vectors(img, polygons, out_path):
    canvas = img.copy()
    h, w = canvas.shape[:2]
    
    # Overly dim the background to make vectors pop
    canvas = (canvas * 0.4).astype(np.uint8)

    region_mask = np.zeros((h, w), dtype=np.uint8)
    kernel_mask = np.zeros((h, w), dtype=np.uint8)

    for poly in polygons:
        poly_pts = np.array(poly, dtype=np.int32)
        
        # 1. Text Region
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [poly_pts], 255)
        region_mask = cv2.bitwise_or(region_mask, poly_mask)
        
        # 2. Text Kernel (Shrunk by Morphological Erosion to simulate cores)
        # Using distance transform for shrinking
        dist = cv2.distanceTransform(poly_mask, cv2.DIST_L2, 3)
        max_dist = np.max(dist)
        # Shrink to 30% of max distance (core)
        core = np.zeros_like(poly_mask)
        core[dist > (max_dist * 0.3)] = 255
        kernel_mask = cv2.bitwise_or(kernel_mask, core)

        # Draw boundaries
        cv2.polylines(canvas, [poly_pts], isClosed=True, color=(0, 165, 255), thickness=2)
        
        # Draw Kernels as faint blue fill
        blue_fill = np.zeros_like(canvas)
        blue_fill[core > 0] = (255, 0, 0)
        canvas = cv2.addWeighted(canvas, 1.0, blue_fill, 0.4, 0)

    # 3. Dense Vector Fields (Centripetal Shifts)
    # Calculate Distances from every pixel to the nearest Kernel boundary
    # Invert Kernel so text core is 0 distance
    dist_to_kernel, labels = cv2.distanceTransformWithLabels(255 - kernel_mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    
    grid = 15
    for y in range(0, h, grid):
        for x in range(0, w, grid):
            if region_mask[y, x] > 0 and kernel_mask[y, x] == 0:
                # We are in the border/shift region
                # Find gradient towards the nearest kernel
                dx = dist_to_kernel[y, min(x+1, w-1)] - dist_to_kernel[y, max(x-1, 0)]
                dy = dist_to_kernel[min(y+1, h-1), x] - dist_to_kernel[max(y-1, 0), x]
                
                length = math.sqrt(dx**2 + dy**2)
                if length > 0:
                    vx, vy = -dx/length, -dy/length # Reverse gradient to point inwards
                    arrow_len = 8
                    pt2 = (int(x + vx * arrow_len), int(y + vy * arrow_len))
                    cv2.arrowedLine(canvas, (x, y), pt2, color=(0, 255, 0), thickness=1, tipLength=0.3)
                    
            elif kernel_mask[y, x] > 0:
                # Inside kernel, draw small red dot to show "core"
                cv2.circle(canvas, (x, y), 1, (0, 0, 255), -1)

    cv2.imwrite(out_path, canvas)
    print(f"Saved Dense Vector Shift visualization to {out_path}")

def main():
    img_dir = r"D:\computer vision\Scene-total-text-detection\dataset\totaltext\train\Images"
    ann_dir = r"D:\computer vision\Scene-total-text-detection\dataset\totaltext\train\annotations"
    
    image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not image_paths:
        print("No images found!")
        return
        
    # Find an image with heavily curved text, usually TotalText img2 or img10 is good
    img_path = image_paths[10] # Try 5th image
    img_id = os.path.basename(img_path).split('.')[0]
    img = cv2.imread(img_path)
    if img is None: return
    
    ann_path = os.path.join(ann_dir, f"poly_gt_{img_id}.txt")
    polygons = parse_txt_annotation(ann_path)
    
    out_dir = r"D:\computer vision\Scene-total-text-detection\output"
    os.makedirs(out_dir, exist_ok=True)
    
    visualize_radial(img, polygons, os.path.join(out_dir, "visualize_radial_rays.jpg"))
    visualize_vectors(img, polygons, os.path.join(out_dir, "visualize_centripetal_vectors.jpg"))

if __name__ == '__main__':
    main()
