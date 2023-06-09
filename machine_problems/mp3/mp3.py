import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():

    # Load in images
    moon_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp3/test_images/moon.bmp', cv2.IMREAD_GRAYSCALE)

    # Preform histogram equalization to improve image quality
    moon_histogram_equalization = histogram_equalization(moon_img)

    # Display images grid
    moon = cv2.hconcat([np.uint8(moon_img), np.uint8(moon_histogram_equalization)])
    # Display
    cv2.imshow(' 1: Normal                                                2: Improved', moon)
    # Wait for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
Image Histogram Equalization

args:
    - image: (cv2.IMREAD_GRAYSCALE) Gray scale image
"""
def histogram_equalization(image):
    # Input Gray Level [0,L1]
    L1 = 255
    # Output Gray Level [0,L2]
    L2 = 255

    # Plot histogram of original image intensities
    image_histogram(image)

    # Histogram intensity data from image
    hist, bins = np.histogram(image, bins=256, range=(0, 256))
    # Calculate cumulative distribution
    cumulative_hist = np.cumsum(hist)
    # Normalize the cumulative values
    cumulative_normalized = cumulative_hist / np.max(cumulative_hist)
    # Plot cumulative distribution
    plt.figure()
    plt.plot(bins[:-1], cumulative_normalized)
    # Set the axis labels and title
    plt.xlabel('Input image pixel intensity')
    plt.ylabel('Output image pixel intensity normalized')
    plt.title('Cumulative histogram distribution')

    # Copy image
    histogram_equalization_image = image.copy()
    # Get image dimensions
    height, width = histogram_equalization_image.shape
    # Preform histogram equalization / Transfer function
    for u in range(height):
        for v in range(width):
            histogram_equalization_image[u,v] = cumulative_normalized[histogram_equalization_image[u,v]]*L2
    # Plot histogram of new histogram equalization image
    image_histogram_equalization(histogram_equalization_image)

    # Show plots
    plt.show()

    return histogram_equalization_image

"""
Plot histogram of original image

args:
    - image: (cv2.IMREAD_GRAYSCALE) Gray scale image
"""
def image_histogram(image):
    # Histogram of pixel intensities
    histogram, bins = np.histogram(image.ravel(), bins=256, range=[0, 256])

    # Plot the histogram
    plt.figure()
    plt.plot(histogram, color='black')
    # Set the axis labels and title
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Original Histogram of image intensities')

"""
Plot histogram of new histogram equalization image

args:
    - image: (cv2.IMREAD_GRAYSCALE) Gray scale image
"""
def image_histogram_equalization(image):
    # Histogram of pixel intensities
    histogram2, bins2 = np.histogram(image.ravel(), bins=256, range=[0, 256])

    # Plot the histogram
    plt.figure()
    plt.plot(histogram2, color='black')
    # Set the axis labels and title
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of histogram equalization image intensities')

if __name__ == '__main__':
    main()