import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():

    # Load in images
    moon_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp3/test_images/moon.bmp', cv2.IMREAD_GRAYSCALE)




    # Plot histogram of original image
    image_histogram(moon_img)

    # Preform histogram equalization to improve image quality
    histogram_equalization(moon_img)



    # Display images grid
    moon = cv2.hconcat([np.uint8(moon_img), np.uint8(moon_img)])

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
    # Show plots
    plt.show()

    print(f"\n cumulative_hist = {cumulative_normalized}")



"""
Plot histogram of original image

args:
    - image: (cv2.IMREAD_GRAYSCALE) Gray scale image
"""
def image_histogram(image):
    # Histogram of pixel intensities
    histogram, bins = np.histogram(image.ravel(), bins=256, range=[0, 256])

    # Plot the histogram
    plt.plot(histogram, color='black')
    # Set the axis labels and title
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of moon image')
    # Show the plot
    # plt.show()


if __name__ == '__main__':
    main()