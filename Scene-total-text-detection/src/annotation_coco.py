import numpy as np


def resample_polygon(vertices, target_n=14):
    vertices = np.array(vertices, dtype=float)

    # close polygon
    vertices = np.vstack([vertices, vertices[0]])

    edges = vertices[1:] - vertices[:-1]
    lengths = np.sqrt((edges ** 2).sum(axis=1))

    perimeter = lengths.sum()
    step = perimeter / target_n

    points = []
    dist_acc = 0
    i = 0

    points.append(vertices[0])

    while len(points) < target_n:
        if dist_acc + lengths[i] >= step:
            ratio = (step - dist_acc) / lengths[i]
            new_point = vertices[i] + ratio * edges[i]
            points.append(new_point)

            vertices[i] = new_point
            edges[i] = vertices[i + 1] - vertices[i]
            lengths[i] = np.linalg.norm(edges[i])
            dist_acc = 0
        else:
            dist_acc += lengths[i]
            i += 1

    return np.array(points)


def coco_mask_to_vertices(mask):
    """
    Convert COCO-Text mask list to Nx2 vertex array
    """
    mask = np.array(mask).reshape(-1, 2)
    return mask


def vertices_to_coco_mask(vertices):
    """
    Convert Nx2 vertices back to COCO-style flat list
    """
    return vertices.flatten().tolist()


def upsample_coco_polygon(mask, target_vertices=14):
    """
    Full pipeline:
    COCO mask → vertices → resample → flattened mask
    """

    vertices = coco_mask_to_vertices(mask)

    resampled = resample_polygon(vertices, target_vertices)

    return vertices_to_coco_mask(resampled)

def process_cocotext(input_file, output_file):
    import json
    import sys
    
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("Processing annotations...")
    count = 0
    errors = 0
    for ann_id, ann in data.get('anns', {}).items():
        if 'mask' in ann and isinstance(ann['mask'], list) and len(ann['mask']) > 0:
            mask = ann['mask']
            try:
                if isinstance(mask[0], list):
                    # In case mask is a list of lists 
                    if len(mask[0]) >= 6:
                        new_mask = upsample_coco_polygon(mask[0], target_vertices=14)
                        ann['mask'] = [new_mask]
                        count += 1
                else:
                    # Flat list
                    if len(mask) >= 6:
                        new_mask = upsample_coco_polygon(mask, target_vertices=14)
                        ann['mask'] = new_mask
                        count += 1
            except Exception as e:
                errors += 1
                
    print(f"Upsampled {count} annotations with 14 points.")
    if errors > 0:
        print(f"Skipped {errors} annotations due to errors.")
    
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    print("Done!")

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3:
        process_cocotext(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python annotation_coco.py <input.json> <output.json>")
