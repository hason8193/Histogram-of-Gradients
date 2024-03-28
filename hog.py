import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import io, exposure, color

image = io.imread("tesla.jpg")

if len(image.shape) > 2:
    image = color.rgb2gray(image)

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, feature_vector=False)

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

plt.subplot(1, 2, 2)
plt.imshow(hog_image_rescaled, cmap='gray')
plt.show()
