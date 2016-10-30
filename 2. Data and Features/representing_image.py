from scipy import misc

img = misc.imread('Datasets/image.jpg')

print(type(img))                    # numpy.ndarray
print(img.shape, img.dtype)         # (1080, 1920, 3), dtype('uint8')

img = img[::2, ::2]                 # Shrink image for faster computing
img = (img / 255.0).reshape(-1, 3)  # Scale colours to 0-1

# Grayscale - Luminance formula
red = img[:,0]
green = img[:,1]  
blue = img[:,2]
gray = (0.299*red + 0.587*green + 0.114*blue)   # Better for machine learning

print(img.shape)
print(gray.shape)