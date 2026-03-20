import os
import re
import numpy as np


def resample_polygon(vertices, target_n=14):
    vertices = np.array(vertices, dtype=float)

    # close polygon
    vertices = np.vstack([vertices, vertices[0]])

    edges = vertices[1:] - vertices[:-1]
    lengths = np.sqrt((edges ** 2).sum(axis=1))

    perimeter = lengths.sum()
    if perimeter == 0:
        return vertices[:-1]
        
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
            if i >= len(lengths):
                break

    # Fix edge case where float precision issue causes loop to end at target_n - 1 points
    while len(points) < target_n:
        points.append(vertices[-1])
        
    return np.array(points)


def process_txt_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    modified = False
    new_lines = []

    # Regex to match the typical Total-Text annotation line format
    # Example format: x: [[214 280 362 349 284 231]], y: [[325 290 320 347 316 346]], ornt: [u'c'], transcriptions: [u'ASRAMA']
    pattern = re.compile(
        r"x: \[\[(.*?)\]\],\s*y: \[\[(.*?)\]\],\s*(ornt:.*?transcriptions:.*)"
    )

    for line in lines:
        line = line.strip()
        if not line:
            new_lines.append(line + "\n")
            continue

        match = pattern.search(line)
        if match:
            x_str = match.group(1)
            y_str = match.group(2)
            rest_of_line = match.group(3)

            try:
                # TotalText format inside brackets is space-separated
                x_vals = [float(v) for v in x_str.replace(',', ' ').split() if v.strip()]
                y_vals = [float(v) for v in y_str.replace(',', ' ').split() if v.strip()]

                if len(x_vals) == len(y_vals) and len(x_vals) > 0:
                    vertices = np.column_stack((x_vals, y_vals))

                    # Resample to exactly 14 vertices
                    resampled = resample_polygon(vertices, target_n=14)
                    
                    # Convert back to integers
                    resampled_int = np.round(resampled).astype(int)

                    # Create new space-separated coordinate strings
                    new_x_str = " ".join(str(v[0]) for v in resampled_int)
                    new_y_str = " ".join(str(v[1]) for v in resampled_int)

                    # Reconstruct line
                    new_line = f"x: [[{new_x_str}]], y: [[{new_y_str}]], {rest_of_line}"
                    new_lines.append(new_line + "\n")
                    modified = True
                else:
                    new_lines.append(line + "\n")
            except Exception as e:
                print(f"Error parsing line in {filepath}: {line}")
                new_lines.append(line + "\n")
        else:
            new_lines.append(line + "\n")

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
            
    return modified


def process_directory(directory):
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return 0
        
    count = 0
    filepaths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
    
    for filepath in filepaths:
        if process_txt_file(filepath):
            count += 1
            
    print(f"Processed/Modified {count} files in {directory}")
    return count


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 2:
        for p in sys.argv[1:]:
            print(f"Processing {p}...")
            process_directory(p)
    else:
        print("Usage: python annotation_totaltext.py <dir_path1> <dir_path2> ...")
