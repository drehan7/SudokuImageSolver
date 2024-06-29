#!/usr/bin/env python3

import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure

im = plt.imread("num1.png")
print("SIZE: ", im.shape)
fd, hog_image = hog(im, orientations=8, pixels_per_cell=(5,5), cells_per_block=(1,1), visualize=True, channel_axis=False)
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0,10))

plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.imshow(im, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(hog_image_rescaled, cmap='gray')

