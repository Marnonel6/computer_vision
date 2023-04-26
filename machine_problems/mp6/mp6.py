"""
ABCD

Author: Marthinus (Marno) Nel
Date: 04/24/2023
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():

    # Load in images as BGR
    test_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp6/test_images/test.bmp', cv2.IMREAD_COLOR)
    test2_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp6/test_images/test2.bmp', cv2.IMREAD_COLOR)
    input_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp6/test_images/input.bmp', cv2.IMREAD_COLOR)

    # Edge detection with Sobel
    test_img_magnitude, test_img_direction = Sobel(test_img, 100)
    test2_img_magnitude, test2_img_direction = Sobel(test2_img, 100)
    input_img_magnitude, input_img_direction = Sobel(input_img, 100)

    # Hough transform
    test_img_hough = HoughTransform(test_img_magnitude, 0.5)
    test2_img_hough = HoughTransform(test2_img_magnitude, 0.5)
    input_img_hough = HoughTransform(input_img_magnitude, 0.5)

    # Filter to only keep higher votes
    test_img_hough[test_img_hough < 120] = 0
    test2_img_hough[test2_img_hough < 120] = 0
    input_img_hough[input_img_hough < 120] = 0

    # Filter out zero values
    test_img_hough_non_zero = test_img_hough[test_img_hough != 0]
    test2_img_hough_non_zero = test2_img_hough[test2_img_hough != 0]
    input_img_hough_non_zero = input_img_hough[input_img_hough != 0]

    # Print maximum, minimum, median and mean of non-zero values
    print('Test Image: Max: {}, Min: {}, Median: {}, Mean: {}'.format(np.max(test_img_hough_non_zero), np.min(test_img_hough_non_zero), np.median(test_img_hough_non_zero), np.mean(test_img_hough_non_zero)))
    print('Test2 Image: Max: {}, Min: {}, Median: {}, Mean: {}'.format(np.max(test2_img_hough_non_zero), np.min(test2_img_hough_non_zero), np.median(test2_img_hough_non_zero), np.mean(test2_img_hough_non_zero)))
    print('Input Image: Max: {}, Min: {}, Median: {}, Mean: {}'.format(np.max(input_img_hough_non_zero), np.min(input_img_hough_non_zero), np.median(input_img_hough_non_zero), np.mean(input_img_hough_non_zero)))

    # Display images at each step in Hough transform
    plt.figure(1)
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    plt.title('Input')
    plt.subplot(2,3,2)
    plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(test_img_magnitude), cv2.COLOR_GRAY2RGB))
    plt.title('Sobel Edge')
    plt.subplot(2,3,3)
    plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(test_img_hough), cv2.COLOR_GRAY2RGB))
    plt.xlabel('rho')
    plt.ylabel('theta [Scaled with ratio]')
    plt.title('Hough Transform')

    plt.figure(2)
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(test2_img, cv2.COLOR_BGR2RGB))
    plt.title('Test2 Image')
    plt.subplot(2,3,2)
    plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(test2_img_magnitude), cv2.COLOR_GRAY2RGB))
    plt.title('Sobel Edge')
    plt.subplot(2,3,3)
    plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(test2_img_hough), cv2.COLOR_GRAY2RGB))
    plt.xlabel('rho')
    plt.ylabel('theta [Scaled with ratio]')
    plt.title('Hough Transform')

    plt.figure(3)
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    plt.title('Input Image')
    plt.subplot(2,3,2)
    plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(input_img_magnitude), cv2.COLOR_GRAY2RGB))
    plt.title('Sobel Edge')
    plt.subplot(2,3,3)
    plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(input_img_hough), cv2.COLOR_GRAY2RGB))
    plt.xlabel('rho')
    plt.ylabel('theta [Scaled with ratio]')
    plt.title('Hough Transform')

    plt.show()


"""
Edge detection with Sobel

args:
    - img: (cv2 - gray image) image to be processed
    - threshold: (int) threshold for edge detection [Lower threshold more edges and more noise]
return:
    - magnitude: (cv2 - gray image) gradient magnitude (Displays edges)
    - direction: (cv2 - gray image) gradient direction
"""
def Sobel(img, threshold):
    # Convert BGR to Gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply sobel filter to image
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitude
    magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    # Calculate gradient direction
    direction = np.arctan2(sobel_y, sobel_x)

    # Normalize gradient magnitude
    magnitude *= 255.0 / magnitude.max()

    # Threshold gradient magnitude
    magnitude[magnitude < threshold] = 0

    return magnitude, direction

"""
Use Sobel edge image to vote for lines in polar space (rho, theta)(Parameter space)

args:
    - img: (cv2 - gray image) image with edges detected from Sobel or other edge detectors
    - threshold: (float) threshold for voting [Lower threshold more edges and more noise]
return:
    - predicted_lines: (cv2 - gray image) voted lines in polar space (rho, theta)
"""
def HoughTransform(img, threshold=0.5):
    # NOTE Debug
    print("Start!")

    # Get image dimensions
    row, col = img.shape

    # Calculate max rho [-sqrt(R^2 + C^2) -> sqrt(R^2 + C^2)]
    max_rho = np.sqrt(row*row + col*col)
    min_rho = -np.sqrt(row*row + col*col)
    # Max theta [-pi/2, pi/2]
    max_theta = np.pi/2
    min_theta = -np.pi/2

    # Scale theta to be represented in the same size as rho
    ratio = max_rho/max_theta
    # Scale factor for precision - Higher more precision and more computation time
    precision_scale = 3

    # Initialize the maximum size of the polar space as the range min to max of theta and rho
    polar_space_voting = np.zeros((int(max_theta*2*ratio*precision_scale), int(max_rho*2*precision_scale)))

    # Loop through image and vote for lines
    for y in range(row):
        for x in range(col):
            if img[y, x] > threshold:
                # for theta in range(int(min_theta), int(max_theta)):
                for theta in np.arange(min_theta, max_theta-0.1, 0.001):
                    rho = x * np.cos(theta) + y * np.sin(theta)
                    polar_space_voting[int((theta + max_theta)*ratio*precision_scale), int((rho + max_rho)*precision_scale)] += 1 # NOTE to make axis positive and not to -pi/2

    # NOTE Debug
    print("Done!")

    """ Scaling for clearer parameter display. Choose 1 or 2"""
    """ 1 """
    # # NOTE Scaling to 255 used to make image display better
    # # Scale polar_space_voting intensity to have values between 55 and 255
    # polar_space_voting *= 200.0 / polar_space_voting.max()
    # # Add 50 if pixel value does not equal 0 to increase visibility of all pixels
    # polar_space_voting[polar_space_voting != 0] += 55
    # NOTE Scaling to 255 used to make image display better
    # Scale polar_space_voting intensity to have values between 100 and 255
    polar_space_voting *= 155.0 / polar_space_voting.max()
    # Add 50 if pixel value does not equal 0 to increase visibility of all pixels
    polar_space_voting[polar_space_voting != 0] += 100
    """ 2 """
    # # NOTE histogram_equalization used to make image display better
    # polar_space_voting *= 255.0 / polar_space_voting.max()
    # histogram_equalization(cv2.convertScaleAbs(polar_space_voting))

    return polar_space_voting

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
    plt.show()

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
