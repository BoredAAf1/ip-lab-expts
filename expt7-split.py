import numpy as np
from PIL import Image, ImageOps, ImageDraw


def split(image,y,x,h,w,threshold,min_size, regions=None):
    if regions is None:
        regions = []

    region = image[y:y+h, x:x+w]

    if np.var(region) < threshold or h <= min_size or w <= min_size:
        mean_val = np.mean(region)
        regions.append((y, x, h, w, mean_val))
        return 
    
    h2, w2 = h//2, w//2
    split(image, y,x,h2,w2, threshold, min_size, regions)
    split(image, y+h2,x,h2,w2, threshold, min_size, regions)
    split(image, y,x+w2,h2,w2, threshold, min_size, regions)
    split(image, y+h2,x+w2,h2,w2, threshold, min_size, regions)

def merge(regions, threshold):
    skip = set()
    merged = []

    n = len(regions)

    for i in range(n):
        if i in skip:
            continue

        merged_flag = False 
        y1, x1, h1, w1, v1 = regions[i]

        for j in range(i + 1, n):
            if j in skip:
                continue

            y2, x2, h2, w2, v2 = regions[j]

            h_adj = (y1 == y2 and h1 == h2 and (x1 + w1 == x2 or x2 + w2 == x1))
            v_adj = (x1 == x2 and w1 == w2 and (y1 + h1 == y2 or y2 + h2 == y1))

            if h_adj or v_adj:
                if abs(v1 - v2) < threshold:
                    v = (v1 + v2) // 2
                    new_y = min(y1, y2)
                    new_x = min(x1, x2)

                    if h_adj:
                        new_h = h1 
                        new_w = w1 + w2 
                    else:
                        new_h = h1 + h2 
                        new_w = w1 

                    skip.add(i)
                    skip.add(j)
                    merged.append((new_y, new_x, new_h, new_w, v))
                    merged_flag = True 
                    break 
        
        if not merged_flag:
            merged.append(regions[i])

    return merged

def split_and_merge(image, split_thresh, merge_thresh, min_size):
    H, W = image.shape
    regions = []
    split(image, 0, 0, H, W, split_thresh, min_size, regions)
    
    merged = regions
    while True:
        new = merge(regions, merge_thresh)
        if len(merged) == len(new):
            break 
        merged = new 

    segmented = np.zeros_like(image)
    for y,x,h,w,v in merged:
        y_end = min(y+h, H)
        x_end = min(x+w, W)
        segmented[y:y_end, x:x_end] = v 

    return segmented, merged 


image = Image.open('./images/binary.png').convert('L')
image = np.array(image)
segmented, merged = split_and_merge(image, 5, 100, 5)

seg = Image.fromarray(segmented).convert("RGB")
draw = ImageDraw.Draw(seg)

for y,x,h,w,_ in merged:
    draw.rectangle([x,y,x+w,y+h], outline="red", width=1)

seg.save('./images/op/segmented.png')