import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from multiprocessing.pool import ThreadPool

def build_filters(ksize, sigma, a, b, c):
    filters = []
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, a, b, c, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters

def display_filter(ksize, sigma, a, b, c, fi):
    filters = build_filters(ksize, sigma, a, b, c)
    plt.imshow(filters[fi], cmap=cm.Greys_r)
    plt.title("Gabor Filter (theta={})".format(fi * 22.5))
    plt.axis("off")
    plt.show()

def build_gfilters(ksize, sigma):
    filters = []
    kern = cv2.getGaussianKernel(ksize, sigma, ktype=cv2.CV_32F)
    k2 = kern * kern.T
    filters.append(k2)
    return filters

def process_threaded(img, filters, threadn=8):
    def f(kern):
        return cv2.matchTemplate(img, kern, cv2.TM_CCORR_NORMED)
    
    pool = ThreadPool(processes=threadn)
    accum = None
    for fimg in pool.imap_unordered(f, filters):
        if accum is None:
            accum = np.zeros_like(fimg)
        accum += fimg * fimg
    return accum

if __name__ == '__main__':
    # Load the image.
    img_fn = "./endoscopic images/image.jpg"
    img = cv2.imread(img_fn)
    
    # Convert the image to grayscale and float32 format.
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gimg = gimg.astype(np.float32) / 255.0
    
    # Build the filter banks.
    gabor_filters = build_filters(127, 15.0, 127.0, 1.0, 0.5)
    gauss_filters = build_gfilters(127, 31.0)
    
    # Apply the filters and get the responses.
    gabor_responses = process_threaded(gimg, gabor_filters)
    gauss_responses = process_threaded(gimg, gauss_filters)
    
    # Display the original image.
    cv2.imshow("Original Image", img)
    
    # Display the responses of both filter banks.
    plt.subplot(1, 2, 1)
    plt.imshow(gabor_responses, cmap=cm.Greys_r)
    plt.title("Gabor Filter Responses")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(gauss_responses, cmap=cm.Greys_r)
    plt.title("Gaussian Filter Responses")
    plt.axis("off")
    
    plt.show()
    
    # Close all windows.
    cv2.destroyAllWindows()
