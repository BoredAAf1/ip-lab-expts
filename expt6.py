import numpy as np
from PIL import Image 
from collections import deque 


class MorphologicalOperations:
    """Image is binary (0/255) numpy array""" 
    def erosion(self, image, k=3):
        p = k//2
        padded_img = np.pad(image, ((p, p), (p, p)), mode='edge')
        eroded_image = np.zeros_like(image)

        for i in range(eroded_image.shape[0]):
            for j in range(eroded_image.shape[1]):
                part = padded_img[i:i+k, j:j+k]
                eroded_image[i, j] = np.min(part)

        return eroded_image

    def dilation(self, image, k=3):
        p = k//2
        padded_img = np.pad(image, ((p, p), (p, p)), mode='edge')
        dilated = np.zeros_like(image)

        for i in range(dilated.shape[0]):
            for j in range(dilated.shape[1]):
                part = padded_img[i:i+k, j:j+k]
                dilated[i, j] = np.max(part)

        return dilated

    def opening(self, image, k=3):
        return self.dilation(self.erosion(image, k=k), k=k)

    def closing(self, image, k=3):
        return self.erosion(self.dilation(image, k=k), k=k)

    def boundary_extraction(self, image, k=3):
        return image - self.erosion(image, k=k)

    def skeletonisation(self, image, k=3):
        temp = image.copy()
        skel = np.zeros_like(image)

        for i in range(10):
            print(f"Iteration {i}")
            eroded = self.erosion(temp, k=k)
            opened = self.opening(eroded, k=k)
            diff = eroded - opened          
            skel = skel | diff    

            temp = eroded
            if not eroded.any():   # stop when empty
                break

        return skel
        
    def remove_small_objects(self, image, thresh=50):
        h, w = image.shape
        
        # ensure binary 0/255
        image = (image > 0).astype(np.uint8) * 255

        def n8(y, x):
            neighbours = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx

                    if 0 <= ny < h and 0 <= nx < w:
                        neighbours.append((ny, nx))

            return neighbours

        visited = set()

        def dfs(y, x):
            stack = [(y, x)]
            comp = []

            while stack:
                cy, cx = stack.pop()
                if (cy, cx) in visited:
                    continue

                visited.add((cy, cx))
                comp.append((cy, cx))

                for ny, nx in n8(cy, cx):
                    if image[ny, nx] != 0 and (ny, nx) not in visited:
                        stack.append((ny, nx))

            return comp 

        new_image = np.zeros_like(image)

        for y in range(h):
            for x in range(w):
                if image[y, x] != 0 and (y, x) not in visited:
                    comp = dfs(y, x)
                    print(len(comp))
                    if len(comp) > thresh:
                        for py, px in comp:
                            new_image[py, px] = 255

        return new_image


image = Image.open('./images/binary.png').convert('L')
image = np.array(image)
image = (image > 128).astype(int) * 255 
print(np.unique(image))

morpho = MorphologicalOperations()

res = morpho.remove_small_objects(image, thresh=900)
Image.fromarray(res).show()