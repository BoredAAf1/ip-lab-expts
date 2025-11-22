import numpy as np
from PIL import Image 
import random 
from collections import deque 
import cv2 

def flood_fill(image, labels, current_label, start_y, start_x, tolerance):
    # BFS, mark only pixels as same component if values < tolereance 
    queue = deque([(start_y, start_x)])
    labels[start_y, start_x] = current_label
    h,w = image.shape
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)] 

    while queue:
        cy, cx = queue.popleft()
        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx 
            if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] == -1:
                if abs(image[ny, nx] - image[cy, cx]) < tolerance:
                    labels[ny, nx] = current_label
                    queue.append((ny, nx))
    
def watershed_segmentation(gradient, tolerance):
    h, w = gradient.shape 
    labels = -np.ones_like(gradient, dtype=np.int32)
    current_label = 0

    sorted_grad = np.argsort(gradient.ravel())

    for index in sorted_grad:
        y = index // w 
        x = index % w 

        if labels[y, x] == -1:
            flood_fill(gradient, labels, current_label, y, x, tolerance)
            current_label += 1 
    
    return labels, current_label

image = Image.open('./images/binary.png').convert('L')
image = np.array(image)
gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
grad = cv2.magnitude(gx, gy)
grad = grad.astype(np.float32)

labels, num_labels = watershed_segmentation(grad, 10)

col_map = [(0, 0, 0)]   # 0 for all background images  
random.seed(42)
for _ in range(num_labels):
    r = random.randint(50, 255)
    g = random.randint(50, 255)
    b = random.randint(50, 255)
    col_map.append((r,g,b))

result_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
for y in range(labels.shape[0]):
    for x in range(labels.shape[1]):
        result_image[y, x] = col_map[labels[y, x]]
    
Image.fromarray(result_image, "RGB").show()

# if want boundary
# overlay = np.zeros((H, W, 3), dtype=np.uint8)

# for y in range(H):
#     for x in range(W):
#         is_boundary = False
#         current = labels[y, x]

#         for dy, dx in [(0,1),(0,-1),(1,0),(-1,0)]:
#             ny, nx = y + dy, x + dx
#             if 0 <= ny < H and 0 <= nx < W:
#                 if labels[ny, nx] != current:
#                     is_boundary = True
#                     break

#         if is_boundary:
#             overlay[y, x] = (255, 0, 0)      
#         else:
#             v = img_arr[y, x]
#             overlay[y, x] = (v, v, v)        

# Image.fromarray(overlay, "RGB").show()

        