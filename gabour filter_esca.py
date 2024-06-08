import cv2
import numpy as np
import matplotlib.pyplot as plt

def deg_to_rad(degree):
    """Convert degrees to radians."""
    return 2 * np.pi / 360 * degree

# Set the parameters for the Gabor filter.
ksize = (21, 21)  # Size of Gabor filter.
sigma = 8.0  # Standard deviation of the Gaussian function.
theta = deg_to_rad(0)  # Orientation of the normal to the parallel stripes.
lamda = np.pi / 2  # Wavelength of the sinusoidal factor.
gamma = 0.5  # Spatial aspect ratio.
phi = 0  # Phase offset.
ktype = cv2.CV_64F  # Type and range of values that each pixel in the Gabor kernel can hold.

# Load the image.
image = cv2.imread("image.jpg")

# Convert the image to grayscale.
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a Gabor filter kernel.
g_kernel = cv2.getGaborKernel(ksize, sigma, theta, lamda, gamma, phi, ktype)

# Filter the image with the Gabor filter.
filtered_img = cv2.filter2D(gray_img, cv2.CV_8UC3, g_kernel)

# Display the filtered image.
cv2.imshow("Filtered Image", filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

