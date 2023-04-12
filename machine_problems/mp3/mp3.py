import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():

    # Load in images
    moon_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp3/test_images/moon.bmp', cv2.IMREAD_GRAYSCALE)

    # Histogram of pixel intensities
    histogram, bins = np.histogram(moon_img.ravel(), bins=256, range=[0, 256])

    # Plot the histogram
    plt.plot(histogram, color='black')
    # Set the axis labels and title
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of moon image')
    # Show the plot
    plt.show()

    # Display images grid
    moon = cv2.hconcat([np.uint8(moon_img), np.uint8(moon_img)])

    # Display
    cv2.imshow(' 1: Normal                                                2: Improved', moon)

    # Wait for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
<description>

args:
    - <arg_name>: (type) <description>
"""



if __name__ == '__main__':
    main()