import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
img_pil = Image.open("image.png")
plt.imshow(img_pil)
plt.title("original pil image")
plt.axis("off")
plt.show()
img_pil_L = img_pil.convert("L")
print("Pillow image size", img_pil.size)
plt.imshow(img_pil_L, cmap = "grey")
plt.title("grey pil image")
plt.axis("off")
plt.show()
#resized img
plt.imshow(img_pil.resize((200,100)))
plt.title("resized pil image")
plt.axis("off")
plt.show()
#Pillow (startX, startY, endX, endY)
startX = 50
startY = 50
endX = 200
endY = 200
crop_pil = img_pil.crop((startX, startY, endX, endY))
plt.imshow(crop_pil)
plt.title("cropped pil image")
plt.axis("off")
plt.show()
plt.imshow(img_pil.transpose(Image.FLIP_LEFT_RIGHT))
plt.title("original pil image")
plt.axis("off")
plt.show()
plt.imshow(img_pil.transpose(Image.FLIP_TOP_BOTTOM))
plt.title("original pil image")
plt.axis("off")
plt.show()
#img.save("img.png")
img_cv = cv2.imread("image.png")
img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
print("opencv image size", img_cv.shape)
plt.imshow(img_cv_rgb)
plt.title("original cv image")
plt.axis("off")
plt.show()
img_cv_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
print("opencv greyscale image")
plt.imshow(img_cv_gray, cmap = "gray")
plt.title("grey opencv image")
plt.axis("off")
plt.show()
#resized img
plt.imshow(cv2.resize(img_cv_rgb, (200, 100)))
plt.title("resized opencv image")
plt.axis("off")
plt.show()
#resized img
img_crop_cv = img_cv_rgb[startY:endY,startX:endX]
plt.imshow(img_crop_cv)
plt.title("cropped cv image")
plt.axis("off")
plt.show()
plt.imshow(cv2.flip(img_cv_rgb, 1))#horiz flip
plt.title("cropped cv image")
plt.axis("off")
plt.show()
plt.imshow(cv2.flip(img_cv_rgb, 0))#vert flip
plt.title("cropped cv image")
plt.axis("off")
plt.show()
#cv2.imwrite('path/to/save.jpg', image, params=None) (OpenCV)