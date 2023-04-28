"""
Hough transform: Algorithm for line detection

Author: Marthinus (Marno) Nel
Date: 04/27/2023
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.cluster import KMeans 

def main():

    # NOTE Debug
    print("RUNNING!")

    # Load in images as BGR
    test_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp6/test_images/test.bmp', cv2.IMREAD_COLOR)
    test2_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp6/test_images/test2.bmp', cv2.IMREAD_COLOR)
    input_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp6/test_images/input.bmp', cv2.IMREAD_COLOR)

    """ Edge detection with Sobel """
    test_img_magnitude, test_img_direction = Sobel(test_img, 100)
    test2_img_magnitude, test2_img_direction = Sobel(test2_img, 100)
    input_img_magnitude, input_img_direction = Sobel(input_img, 50)

    """ Hough transform """
    test_img_hough, ratio_test, preci_scale_test, max_rho_test, max_theta_test = \
        HoughTransform(test_img_magnitude)
    test2_img_hough, ratio_test2, preci_scale_test2, max_rho_test2, max_theta_test2 = \
        HoughTransform(test2_img_magnitude)
    input_img_hough, ratio_input, preci_scale_input, max_rho_input, max_theta_input = \
        HoughTransform(input_img_magnitude)

    """ K-Means clustering """
    # Filter to only keep higher votes/clusters
    test_img_hough_filter = copy.deepcopy(test_img_hough)
    test2_img_hough_filter = copy.deepcopy(test2_img_hough)
    input_img_hough_filter = copy.deepcopy(input_img_hough)
    test_img_hough_filter[test_img_hough_filter < 100] = 0
    test2_img_hough_filter[test2_img_hough_filter < 100] = 0
    input_img_hough_filter[input_img_hough_filter < 130] = 0
    # get the indices of all non-zero elements
    test_nonzero_indices = np.nonzero(test_img_hough_filter)
    test2_nonzero_indices = np.nonzero(test2_img_hough_filter)
    input_nonzero_indices = np.nonzero(input_img_hough_filter)
    # create a list of (x,y) coordinate pairs from the indices
    test_coordinates = list(zip(test_nonzero_indices[1], test_nonzero_indices[0]))
    test2_coordinates = list(zip(test2_nonzero_indices[1], test2_nonzero_indices[0]))
    input_coordinates = list(zip(input_nonzero_indices[1], input_nonzero_indices[0]))
    # use the coordinates list with sklearn
    test_kmeans = KMeans(n_clusters=4).fit(test_coordinates)
    test2_kmeans = KMeans(n_clusters=6).fit(test2_coordinates)
    input_kmeans = KMeans(n_clusters=5).fit(input_coordinates)
    # get the cluster centroids
    test_centroids = test_kmeans.cluster_centers_
    test2_centroids = test2_kmeans.cluster_centers_
    input_centroids = input_kmeans.cluster_centers_

    """ Draw predicted lines with Cluster centroids [rho, theta] """
    # Create a copy of original image to add predicted lines to
    predicted_lines_test = copy.deepcopy(test_img)
    predicted_lines_test2 = copy.deepcopy(test2_img)
    predicted_lines_input = copy.deepcopy(input_img)
    # Test Image
    for cluster in test_centroids:
        rho = cluster[0]/preci_scale_test-max_rho_test
        theta = (cluster[1]/(max_theta_test*2*ratio_test*preci_scale_test))*np.pi
        for x in range(test_img.shape[0]):
            y = (rho - x*np.cos(theta))/np.sin(theta)
            if y >= 0 and y < test_img.shape[1]:
                predicted_lines_test[x,int(y)] = 255
    # Test2 Image
    for cluster in test2_centroids:
        rho = cluster[0]/preci_scale_test2-max_rho_test2
        theta = (cluster[1]/(max_theta_test2*2*ratio_test2*preci_scale_test))*np.pi
        for x in range(test2_img.shape[0]):
            y = (rho - x*np.cos(theta))/np.sin(theta)
            if y >= 0 and y < test2_img.shape[1]:
                predicted_lines_test2[x,int(y)] = 255
    # Input Image
    for cluster in input_centroids:
        rho = cluster[0]/preci_scale_input-max_rho_input
        theta = (cluster[1]/(max_theta_input*2*ratio_input*preci_scale_test))*np.pi
        for x in range(input_img.shape[0]):
            y = (rho - x*np.cos(theta))/np.sin(theta)
            if y >= 0 and y < input_img.shape[1]:
                predicted_lines_input[x,int(y)] = 255

    # NOTE Debug
    print("FINISHED!")

    # Display images
    Display_images = True
    if Display_images:
        # Display images at each step
        plt.figure(1)
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
        plt.title('Input')
        plt.subplot(2,3,2)
        plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(test_img_magnitude), cv2.COLOR_GRAY2RGB))
        plt.title('Sobel Edge')
        plt.subplot(2,3,3)
        plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(test_img_hough), cv2.COLOR_GRAY2RGB))
        plt.gca().invert_yaxis()
        plt.xlabel('rho')
        plt.ylabel('theta')
        plt.title('Hough Transform')
        plt.subplot(2,3,4)
        plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(test_img_hough_filter), cv2.COLOR_GRAY2RGB))
        plt.gca().invert_yaxis()
        plt.xlabel('rho')
        plt.ylabel('theta')
        plt.title('Hough Transform Filtered')
        plt.subplot(2,3,5)
        plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(test_img_hough_filter), cv2.COLOR_GRAY2RGB))
        plt.scatter(test_centroids[:,0],test_centroids[:,1])
        plt.gca().invert_yaxis()
        plt.xlabel('rho')
        plt.ylabel('theta')
        plt.title('Cluster Centroids [K-means]')
        plt.subplot(2,3,6)
        plt.imshow(predicted_lines_test)
        plt.title('Predicted Lines')

        plt.figure(2)
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(test2_img, cv2.COLOR_BGR2RGB))
        plt.title('Input')
        plt.subplot(2,3,2)
        plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(test2_img_magnitude), cv2.COLOR_GRAY2RGB))
        plt.title('Sobel Edge')
        plt.subplot(2,3,3)
        plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(test2_img_hough), cv2.COLOR_GRAY2RGB))
        plt.gca().invert_yaxis()
        plt.xlabel('rho')
        plt.ylabel('theta')
        plt.title('Hough Transform')
        plt.subplot(2,3,4)
        plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(test2_img_hough_filter), cv2.COLOR_GRAY2RGB))
        plt.gca().invert_yaxis()
        plt.xlabel('rho')
        plt.ylabel('theta')
        plt.title('Hough Transform Filtered')
        plt.subplot(2,3,5)
        plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(test2_img_hough_filter), cv2.COLOR_GRAY2RGB))
        plt.gca().invert_yaxis()
        plt.scatter(test2_centroids[:,0],test2_centroids[:,1])
        plt.xlabel('rho')
        plt.ylabel('theta')
        plt.title('Cluster Centroids [K-means]')
        plt.subplot(2,3,6)
        plt.imshow(predicted_lines_test2)
        plt.title('Predicted Lines')

        plt.figure(3)
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
        plt.title('Input')
        plt.subplot(2,3,2)
        plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(input_img_magnitude), cv2.COLOR_GRAY2RGB))
        plt.title('Sobel Edge')
        plt.subplot(2,3,3)
        plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(input_img_hough), cv2.COLOR_GRAY2RGB))
        plt.gca().invert_yaxis()
        plt.xlabel('rho')
        plt.ylabel('theta')
        plt.title('Hough Transform')
        plt.subplot(2,3,4)
        plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(input_img_hough_filter), cv2.COLOR_GRAY2RGB))
        plt.gca().invert_yaxis()
        plt.xlabel('rho')
        plt.ylabel('theta')
        plt.title('Hough Transform Filtered')
        plt.subplot(2,3,5)
        plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(input_img_hough_filter), cv2.COLOR_GRAY2RGB))
        plt.gca().invert_yaxis()
        plt.scatter(input_centroids[:,0],input_centroids[:,1])
        plt.xlabel('rho')
        plt.ylabel('theta')
        plt.title('Cluster Centroids [K-means]')
        plt.subplot(2,3,6)
        plt.imshow(cv2.cvtColor(predicted_lines_input, cv2.COLOR_BGR2RGB))
        plt.title('Predicted Lines')

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
Calculate Hough Transform: use Sobel edge image to vote for lines in polar space (rho, theta)(Parameter space)

args:
    - img: (cv2 - gray image) image with edges detected from Sobel or other edge detectors
    - threshold: (float) threshold for voting [Lower threshold more edges and more noise]
return:
    - polar_space_voting: (np.array) polar space with votes for lines
    - ratio: (float) ratio between max_rho and max_theta
    - precision_scale: (int) increase or decrease precision
    - max_rho: (float) Maximum value of row in positive axis [sqrt(R^2 + C^2)]
    - max_theta: (float) Maximum value of column in positive axis [pi/2]
"""
def HoughTransform(img, threshold=0.5):
    # Get image dimensions
    row, col = img.shape

    # Calculate max rho [-sqrt(R^2 + C^2) -> sqrt(R^2 + C^2)]
    max_rho = np.sqrt(row*row + col*col)
    min_rho = -np.sqrt(row*row + col*col)
    # Max theta [-pi/2, pi/2]
    max_theta = 90
    min_theta = -90

    # Scale theta to be represented in the same size as rho
    ratio = int(max_rho/max_theta)
    # Scale factor for precision - Higher more precision and more computation time
    precision_scale = 1

    # Initialize the maximum size of the polar space as the range min to max of theta and rho and scaling
    polar_space_voting = np.zeros((int(max_theta*2*ratio)*precision_scale, int(max_rho*2)*precision_scale))

    # Loop through image and vote for lines
    for x in range(row):
        for y in range(col):
            if img[x, y] > threshold:
                for theta in range(0, max_theta*2*ratio*precision_scale):
                    rho = x * np.cos((theta/(max_theta*2*ratio*precision_scale))*np.pi) + \
                          y * np.sin((theta/(max_theta*2*ratio*precision_scale))*np.pi)
                    polar_space_voting[int(theta), int((rho + max_rho)*precision_scale)] += 1

    """ Scaling for clearer parameter display. Choose 1 or 2"""
    """ 1 """
    # NOTE Scaling to 255 used to make image display better
    # # NOTE BETTER ONE below
    # # Scale polar_space_voting intensity to have values between 100 and 255
    # polar_space_voting *= 155.0 / polar_space_voting.max()
    # # Add 100 if pixel value does not equal 0 to increase visibility of all pixels
    # polar_space_voting[polar_space_voting != 0] += 100
    """ 2 """
    # NOTE histogram_equalization used to make image display better
    polar_space_voting *= 255.0 / polar_space_voting.max()
    histogram_equalization(cv2.convertScaleAbs(polar_space_voting))

    return polar_space_voting, ratio, precision_scale, max_rho, max_theta

"""
Histogram equalization
"""
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
    # # Plot cumulative distribution
    # plt.figure()
    # plt.plot(bins[:-1], cumulative_normalized)
    # # Set the axis labels and title
    # plt.xlabel('Input image pixel intensity')
    # plt.ylabel('Output image pixel intensity normalized')
    # plt.title('Cumulative histogram distribution')

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

    # # Show plots
    # plt.show()

    return histogram_equalization_image

"""
Plot histogram of original image

args:
    - image: (cv2.IMREAD_GRAYSCALE) Gray scale image
"""
def image_histogram(image):
    # Histogram of pixel intensities
    histogram, bins = np.histogram(image.ravel(), bins=256, range=[0, 256])

    # # Plot the histogram
    # plt.figure()
    # plt.plot(histogram, color='black')
    # # Set the axis labels and title
    # plt.xlabel('Pixel Intensity')
    # plt.ylabel('Frequency')
    # plt.title('Original Histogram of image intensities')
    # plt.show()

"""
Plot histogram of new histogram equalization image

args:
    - image: (cv2.IMREAD_GRAYSCALE) Gray scale image
"""
def image_histogram_equalization(image):
    # Histogram of pixel intensities
    histogram2, bins2 = np.histogram(image.ravel(), bins=256, range=[0, 256])

    # # Plot the histogram
    # plt.figure()
    # plt.plot(histogram2, color='black')
    # # Set the axis labels and title
    # plt.xlabel('Pixel Intensity')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of histogram equalization image intensities')

if __name__ == '__main__':
    main()
