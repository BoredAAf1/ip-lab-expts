import numpy as np 
from PIL import Image

def split(image, y,x,h,w, threshold, min_size, regions):
    region = image[y:y+h, x:x+w]
    if np.var(region) < threshold or h <= min_size or w <= min_size:
        val = np.mean(region)
        regions.append((y, x, h, w, val))
        return
    
    h2, w2 = h//2, w//2
    h3 = h - h2
    w3 = w - w2

    split(image, y, x, h2, w2, threshold, min_size, regions)
    split(image, y+h2, x, h3, w2, threshold, min_size, regions)
    split(image, y, x+w2, h2, w3, threshold, min_size, regions)
    split(image, y+h2, x+w2, h3, w3, threshold, min_size, regions)

def merge(regions, threshold):
    merged = []
    skip = set()
    
    for i in range(len(regions)):
        if i in skip:
            continue

        is_merged = False
        y1,x1,h1,w1,v1 = regions[i]

        for j in range(i+1, len(regions)):
            if j in skip:
                continue
            y2,x2,h2,w2,v2 = regions[j]

            adj_h = (h1 == h2 and y1 == y2 and (x1 + w1 == x2 or x2 + w2 == x1))
            adj_w = (w1 == w2 and x1 ==x2 and (y1 + h1 == y2 or y2 + h2 == y1))

            if adj_h or adj_w:
                if abs(v1 - v2) < threshold:
                    x = min(x1, x2)
                    y = min(y1, y2)

                    if adj_h:
                        h = h1
                        w = w1 + w2 
                    elif adj_w:
                        h = h1 + h2 
                        w = w1 
                    
                    v = (v1 + v2)//2

                    merged.append((y,x,h,w,v))
                    is_merged = True 
                    skip.add(j)
                    skip.add(i)
    
        if not is_merged:
            merged.append(regions[i])
    
    return merged 

def split_and_merge(image, split_thresh, merge_thresh, min_size):
    regions = []
    H, W = image.shape 
    split(image, 0, 0, H, W, split_thresh, min_size, regions)

    merged = merge(regions, merge_thresh)

    while len(regions) != len(merged):
        regions = merged 
        merged = merge(merged, merge_thresh)
    
    segmented = np.zeros((H, W, 3), dtype=np.uint8)
    for y,x,h,w,v in merged:
        y_end = min(y+h, H)
        x_end = min(x+w, W)
        segmented[y:y_end, x:x_end, :] = v 
        
        segmented[y, x:x_end] = [255, 0, 0]
        segmented[y_end-1, x:x_end] = [255, 0, 0]
        segmented[y:y_end, x] = [255, 0, 0]
        segmented[y:y_end, x_end-1] = [255, 0, 0]

    return segmented

image = Image.open('./images/highway.jpg').convert("L")
img_arr = np.array(image)
segmented = split_and_merge(img_arr, 25, 12, 16)
Image.fromarray(segmented, "RGB").show()

