import numpy as np 
from PIL import Image 

class ArithmeticCoding:
    def get_ranges(self, probs):
        ranges = {}
        low = 0.0 
        for sym, prob in probs.items():
            high = low + prob 
            ranges[sym] = (low, high)
            low = high 
        
        return ranges 

    def encode(self, data):
        unique, counts = np.unique(data, return_counts=True)
        total = len(data)
        probs = {int(sym): count/total for sym, count in zip(unique, counts)}
        ranges = self.get_ranges(probs)

        low = 0.0 
        high = 1.0 
        for symbol in data:     
            sym_low, sym_high = ranges[symbol]
            range_width = high - low 
            # !!! Update high first
            high = low + range_width*sym_high
            low = low + range_width*sym_low
        
        return (low + high) / 2, ranges

    def decode(self, code, message_length, ranges):
        low = 0.0 
        high = 1.0 

        data = []
        for _ in range(message_length):
            range_width = high - low 
            scaled = (code - low) / (range_width + 1e-15)

            for symbol, (sym_low, sym_high) in ranges.items():
                if sym_low <= scaled < sym_high:
                    data.append(symbol)
                    high = low + range_width*sym_high 
                    low = low + range_width*sym_low
                    break 
        
        return data 
    
class ImageArithmeticCoding:
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size
        self.image_height = None 
        self.image_width = None 
        self.ac = ArithmeticCoding()
    
    def read_image(self, path):
        image = Image.open(path).convert('L')
        img_np = np.array(image)
        self.image_height, self.image_width = img_np.shape 
        img_arr = img_np.ravel()
        return img_arr
    
    def encode(self, path):
        print("Encoding....")
        data = self.read_image(path)
        encoded = []
        for i in range(0, len(data), self.chunk_size):
            chunk = data[i:i+self.chunk_size]
            code, ranges = self.ac.encode(chunk)
            encoded.append((code, len(chunk), ranges))

        return encoded 

    def decode(self, encoded):
        print("Decoding...")
        data = []   
        for code, msg_len, ranges in encoded:
            chunk_data = self.ac.decode(code, msg_len, ranges)
            data.extend(chunk_data)
        
        image = np.array(data, dtype=np.uint8).reshape(self.image_height, self.image_width)

        return Image.fromarray(image)

iac = ImageArithmeticCoding(12)
encoded = iac.encode('./images/mario.png')
decoded_img = iac.decode(encoded)
decoded_img.show()