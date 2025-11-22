import numpy as np
from PIL import Image 
import heapq 


class Node: 
    def __init__(self, freq, char = -1, left = None, right = None, huff = ''):
        self.freq = freq 
        self.char = char 
        self.left = left 
        self.right = right 
        self.huff = huff
    
    def __lt__(self, other):
        return self.freq < other.freq 

class TreeBuilder:
    def __init__(self, freq):
        self.pixels = np.arange(0, 256, 1)
        self.freq = freq
        self.root = self.build_tree()

        self.mapping = {}
        self.reverse_mapping = {}
        print("Constructing mapping...")
        self.get_mappings(self.root)


    def build_tree(self):
        print("Building tree....")
        nodes = []
        for i in range(256):
            new_node = Node(self.freq[i], self.pixels[i])
            nodes.append(new_node)

        heapq.heapify(nodes)

        while len(nodes) > 1:
            n1 = heapq.heappop(nodes)
            n2 = heapq.heappop(nodes)
            n1.huff = '0'
            n2.huff = '1'
            n = Node(n1.freq + n2.freq, -1, n1, n2, huff='')

            heapq.heappush(nodes, n)
        
        return nodes[0]

    def get_mappings(self, root, code = ""):
        if not root.left and not root.right:
            self.mapping[root.char] = code 
            self.reverse_mapping[code] = root.char

        if root.left:
            self.get_mappings(root.left, code + root.left.huff)
        
        if root.right:
            self.get_mappings(root.right, code + root.right.huff)
        
    def return_mappings(self):
        return self.mapping, self.reverse_mapping



def compress_image(image_np, mapping):
    print("Compressing image....")
    result = []
    for pxl in image_np.ravel():
        result.append(mapping[pxl])
    
    return "".join(result)

def decompress_image(compressed, reverse_mapping, H, W):
    result = []
    current = []

    for bit in compressed:
        current.append(bit)
        key = ''.join(current)
        if key in reverse_mapping:
            result.append(reverse_mapping[key])
            current.clear()

    return np.array(result, dtype=np.uint8).reshape(H, W)


image = Image.open('./images/mario.png').convert('L')
image_array = np.array(image)
H, W = image_array.shape

freq = np.zeros(256)
for pxl in image_array.ravel():
    freq[pxl] += 1 

tree = TreeBuilder(freq)
mp, rmp = tree.return_mappings()

compressed = compress_image(image_array, mp)

r = len(compressed)
og = image_array.shape[0] * image_array.shape[1] * 8 

print("Compression ratio: ", og / r)

decomp = decompress_image(compressed, rmp, H, W)

Image.fromarray(decomp).show()