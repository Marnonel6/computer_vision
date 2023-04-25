"""
Canny edge detection implemented from scratch

Author: Marthinus (Marno) Nel
Date: 04/24/2023
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import copy

def main():

    # Load in images as BGR
    gun_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp5/test_images/gun1.bmp', cv2.IMREAD_COLOR)
    joy_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp5/test_images/joy1.bmp', cv2.IMREAD_COLOR)
    pointer_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp5/test_images/pointer1.bmp', cv2.IMREAD_COLOR)
    lena_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp5/test_images/lena.bmp', cv2.IMREAD_COLOR)
    test_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp5/test_images/test1.bmp', cv2.IMREAD_COLOR)

    # Preform gaussian smoothing
    smoothed_gun_img = GaussSmoothing(gun_img)
    smoothed_joy_img = GaussSmoothing(joy_img)
    smoothed_pointer_img = GaussSmoothing(pointer_img)
    smoothed_lena_img = GaussSmoothing(lena_img)
    smoothed_test_img = GaussSmoothing(test_img)

    # Compute image gradient
    mag_gun, dir_gun = ImageGradient(smoothed_gun_img)
    mag_joy, dir_joy = ImageGradient(smoothed_joy_img)
    mag_pointer, dir_pointer = ImageGradient(smoothed_pointer_img)
    mag_lena, dir_lena = ImageGradient(smoothed_lena_img)
    mag_test, dir_test = ImageGradient(smoothed_test_img)

    # Compute thresholds
    T_high_gun, T_low_gun = compute_thresholds(mag_gun, ratio=0.4, percentageOfNonEdge=0.85)
    T_high_joy, T_low_joy = compute_thresholds(mag_joy, ratio=0.2, percentageOfNonEdge=0.70)
    T_high_pointer, T_low_pointer = compute_thresholds(mag_pointer, ratio=0.4, percentageOfNonEdge=0.85)
    T_high_lena, T_low_lena = compute_thresholds(mag_lena, ratio=0.2, percentageOfNonEdge=0.75)
    T_high_test, T_low_test = compute_thresholds(mag_test)

    # Apply non-maximum suppression
    nms_gun = NonMaxSuppression(mag_gun, dir_gun)
    nms_joy = NonMaxSuppression(mag_joy, dir_joy)
    nms_pointer = NonMaxSuppression(mag_pointer, dir_pointer)
    nms_lena = NonMaxSuppression(mag_lena, dir_lena)
    nms_test = NonMaxSuppression(mag_test, dir_test)

    # Find strong and weak edges from non-maximum suppression
    mag_high_gun, mag_low_gun, comb_high_low_mag_gun = find_edges(nms_gun, T_high_gun, T_low_gun)
    mag_high_joy, mag_low_joy, comb_high_low_mag_joy = find_edges(nms_joy, T_high_joy, T_low_joy)
    mag_high_pointer, mag_low_pointer, comb_high_low_mag_pointer = find_edges(nms_pointer, T_high_pointer, T_low_pointer)
    mag_high_lena, mag_low_lena, comb_high_low_mag_lena = find_edges(nms_lena, T_high_lena, T_low_lena)
    mag_high_test, mag_low_test, comb_high_low_mag_test = find_edges(nms_test, T_high_test, T_low_test)

    # Edge linking
    edge_linking_gun = EdgeLinking(comb_high_low_mag_gun)
    edge_linking_joy = EdgeLinking(comb_high_low_mag_joy)
    edge_linking_pointer = EdgeLinking(comb_high_low_mag_pointer)
    edge_linking_lena = EdgeLinking(comb_high_low_mag_lena)
    edge_linking_test = EdgeLinking(comb_high_low_mag_test)

    # Display all figures
    display_all =True
    if display_all:

        # Display images at each step in canny edge detection
        plt.figure(1)
        plt.subplot(2, 4, 1)
        plt.imshow(cv2.cvtColor(gun_img, cv2.COLOR_BGR2RGB), cmap='gray')
        plt.title('Original')
        plt.subplot(2, 4, 2)
        plt.imshow(smoothed_gun_img, cmap='gray')
        plt.title('Gaussian Smoothing')
        plt.subplot(2, 4, 3)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(mag_gun), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Image Magnitude')
        plt.subplot(2, 4, 4)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(nms_gun), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Non-Maxima Suppression Gun')
        plt.subplot(2, 4, 5)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(mag_high_gun), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Magnitude strong edges')
        plt.subplot(2, 4, 6)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(mag_low_gun), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Magnitude weak edges')
        plt.subplot(2, 4, 7)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(comb_high_low_mag_gun), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Magnitude strong edges (white), weak (gray)')
        plt.subplot(2, 4, 8)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(edge_linking_gun), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Canny edge detection from scratch')
        plt.show()

        plt.figure(2)
        plt.subplot(2, 4, 1)
        plt.imshow(cv2.cvtColor(joy_img, cv2.COLOR_BGR2RGB), cmap='gray')
        plt.title('Original')
        plt.subplot(2, 4, 2)
        plt.imshow(smoothed_joy_img, cmap='gray')
        plt.title('Gaussian Smoothing')
        plt.subplot(2, 4, 3)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(mag_joy), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Image Magnitude')
        plt.subplot(2, 4, 4)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(nms_joy), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Non-Maxima Suppression Joy')
        plt.subplot(2, 4, 5)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(mag_high_joy), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Magnitude strong edges')
        plt.subplot(2, 4, 6)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(mag_low_joy), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Magnitude weak edges')
        plt.subplot(2, 4, 7)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(comb_high_low_mag_joy), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Magnitude strong edges (white), weak (gray)')
        plt.subplot(2, 4, 8)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(edge_linking_joy), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Canny edge detection from scratch')
        plt.show()

        plt.figure(3)
        plt.subplot(2, 4, 1)
        plt.imshow(cv2.cvtColor(pointer_img, cv2.COLOR_BGR2RGB), cmap='gray')
        plt.title('Original')
        plt.subplot(2, 4, 2)
        plt.imshow(smoothed_pointer_img, cmap='gray')
        plt.title('Gaussian Smoothing')
        plt.subplot(2, 4, 3)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(mag_pointer), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Image Magnitude')
        plt.subplot(2, 4, 4)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(nms_pointer), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Non-Maxima Suppression Pointer')
        plt.subplot(2, 4, 5)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(mag_high_pointer), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Magnitude strong edges')
        plt.subplot(2, 4, 6)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(mag_low_pointer), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Magnitude weak edges')
        plt.subplot(2, 4, 7)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(comb_high_low_mag_pointer), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Magnitude strong edges (white), weak (gray)')
        plt.subplot(2, 4, 8)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(edge_linking_pointer), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Canny edge detection from scratch')
        plt.show()

        plt.figure(4)
        plt.subplot(2, 4, 1)
        plt.imshow(cv2.cvtColor(lena_img, cv2.COLOR_BGR2RGB), cmap='gray')
        plt.title('Original')
        plt.subplot(2, 4, 2)
        plt.imshow(smoothed_lena_img, cmap='gray')
        plt.title('Gaussian Smoothing')
        plt.subplot(2, 4, 3)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(mag_lena), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Image Magnitude')
        plt.subplot(2, 4, 4)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(nms_lena), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Non-Maxima Suppression Lena')
        plt.subplot(2, 4, 5)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(mag_high_lena), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Magnitude strong edges')
        plt.subplot(2, 4, 6)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(mag_low_lena), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Magnitude weak edges')
        plt.subplot(2, 4, 7)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(comb_high_low_mag_lena), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Magnitude strong edges (white), weak (gray)')
        plt.subplot(2, 4, 8)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(edge_linking_lena), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Canny edge detection from scratch')
        plt.show()

        plt.figure(5)
        plt.subplot(2, 4, 1)
        plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB), cmap='gray')
        plt.title('Original')
        plt.subplot(2, 4, 2)
        plt.imshow(smoothed_test_img, cmap='gray')
        plt.title('Gaussian Smoothing')
        plt.subplot(2, 4, 3)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(mag_test), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Image Magnitude')
        plt.subplot(2, 4, 4)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(nms_test), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Non-Maxima Suppression Test')
        plt.subplot(2, 4, 5)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(mag_high_test), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Magnitude strong edges')
        plt.subplot(2, 4, 6)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(mag_low_test), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Magnitude weak edges')
        plt.subplot(2, 4, 7)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(comb_high_low_mag_test), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Magnitude strong edges (white), weak (gray)')
        plt.subplot(2, 4, 8)
        plt.imshow(np.uint8(cv2.cvtColor(cv2.convertScaleAbs(edge_linking_test), cv2.COLOR_GRAY2BGR)), cmap='gray')
        plt.title('Canny edge detection from scratch')
        plt.show()

    """ Test other edge detectors  """
    other_edge_detectors = True
    if other_edge_detectors:
        # Sobel filter
        sobelxy = cv2.Sobel(smoothed_lena_img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

        # Canny filter
        canny = cv2.Canny(lena_img, 100, 200)

        # Roberts filter
        roberts_cross_v = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
        roberts_cross_h = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])
        roberts_v = cv2.filter2D(lena_img, -1, roberts_cross_v)
        roberts_h = cv2.filter2D(lena_img, -1, roberts_cross_h)
        roberts = cv2.addWeighted(roberts_v, 0.5, roberts_h, 0.5, 0)

        # Zero-cross filter
        laplacian = cv2.Laplacian(smoothed_lena_img, cv2.CV_64F)
        thresh = 5
        zero_cross = np.zeros(laplacian.shape, dtype=bool)
        zero_cross[laplacian > thresh] = True
        zero_cross = zero_cross.astype(np.uint8) * 255

        # display results
        cv2.imshow("Original", lena_img)
        cv2.imshow("Sobel", sobelxy)
        cv2.imshow("Canny", canny)
        cv2.imshow("Roberts", roberts)
        cv2.imshow("Zero-Cross", zero_cross)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

"""
Gaussian smoothing

args:
    - image: (cv2 - BGR image) image to apply gaussian smoothing to
    - N: (int) size of the gaussian kernel [NxN]
    - sigma: (float) standard deviation of the gaussian distribution
return:
    - smoothed: (cv2 - gray image) gaussian smoothed image
"""
def GaussSmoothing(image, N=5, sigma=1):
    # Gaussian filter kernel
    size = int(N) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    kernel =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

    # Convolve the kernel with the input image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smoothed = convolve2d(kernel, image)

    return smoothed

"""
Image gradient [Magnitude, Direction]

args:
    - image: (cv2 - gray image) image to apply image gradient to
return:
    - magnitude: (cv2 - gray image) magnitude of the image gradient
    - direction: (cv2 - gray image) direction of the image gradient
"""
def ImageGradient(image):
    # Sobel operators
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Apply Sobel convolution kernels
    Ix = convolve2d(Gx, image)
    Iy = convolve2d(Gy, image)

    # Compute magnitude and direction
    magnitude = np.sqrt(Ix*Ix + Iy*Iy)
    magnitude = (magnitude/magnitude.max()) * 255
    direction = np.arctan2(Iy, Ix)

    return magnitude, direction

"""
Determining high and low thresholds

args:
    - magnitude: (cv2 - gray image) magnitude image for determining the two thresholds
    - ratio: (float) ratio of high to low threshold
    - percentageOfNonEdge: (float) percentage of non-edge pixels
return:
    - high_threshold: (float) high threshold value
    - low_threshold: (float) low threshold value
"""
def compute_thresholds(magnitude, ratio=0.5, percentageOfNonEdge=0.8):

    # Histogram intensity data from magnitude
    hist, bins = np.histogram(magnitude, bins=256, range=(0, 256))
    # Calculate cumulative distribution
    cumulative_hist = np.cumsum(hist)
    # Normalize the cumulative values
    cumulative_normalized = cumulative_hist / np.max(cumulative_hist)
    # Compute high and low thresholds
    high_threshold = np.argwhere(cumulative_normalized >= percentageOfNonEdge)[0][0]
    # print(high_threshold)
    low_threshold = high_threshold * ratio

    return high_threshold, low_threshold

"""
Non-maximum suppression

args:
    - magnitude: (cv2 - gray image) magnitude of the image gradient
    - direction: (cv2 - gray image) direction of the image gradient
return:
    - suppressed: (cv2 - gray image) non-maximum suppressed image
"""
def NonMaxSuppression(magnitude, direction):

    # Suppressed image
    suppressed = np.zeros(magnitude.shape)
    direction = direction * 180 / np.pi
    direction[direction < 0] += 180

    # Loop through the image and find local maxima
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            # Initialize dx and dj as max value
            dx = 255
            dj = 255
            
            # Constrain to matrix direction = 0
            if (0 <= direction[i,j] < 22.5) or (157.5 <= direction[i,j] <= 180):
                dx, dj = magnitude[i, j+1], magnitude[i, j-1]
            # Constrain to matrix direction =  45
            elif (22.5 <= direction[i,j] < 67.5):
                dx, dj = magnitude[i+1, j-1], magnitude[i-1, j+1]
            # Constrain to matrix direction =  90
            elif (67.5 <= direction[i,j] < 112.5):
                dx, dj = magnitude[i+1, j], magnitude[i-1, j]
            # Constrain to matrix direction =  135
            elif (112.5 <= direction[i,j] < 157.5):
                dx, dj = magnitude[i-1, j-1], magnitude[i+1, j+1]

            # Only keep pixel value of two sides are less intense than the current pixel
            if (magnitude[i,j] >= dx) and (magnitude[i,j] >= dj):
                suppressed[i,j] = magnitude[i,j]
            else:
                suppressed[i,j] = 0

    return suppressed

"""
Finds weak and strong edges with high and low threshold

args:
    - suppressed: (cv2 - gray image) non-maximum suppressed image to apply thresholds to
    - T_high: (float) high threshold value
    - T_low: (float) low threshold value
return:
    - strong_edges: (cv2 - gray image) image with strong edges
    - weak_edges: (cv2 - gray image) image with weak edges
    - combined_edges: (cv2 - gray image) image with strong edges (255) and weak edges (125)
"""
def find_edges(suppressed, T_high, T_low):
    # Initialize strong and weak edges
    strong_edges = np.zeros(suppressed.shape)
    weak_edges = np.zeros(suppressed.shape)
    combined_edges = np.zeros(suppressed.shape)

    # Loop through suppressed image
    for i in range(suppressed.shape[0]):
        for j in range(suppressed.shape[1]):
            # Weak edges
            if suppressed[i,j] >= T_low:
                weak_edges[i,j] = 125
                combined_edges[i,j] = 125
            # Strong edges
            if suppressed[i,j] >= T_high:
                strong_edges[i,j] = 255
                combined_edges[i,j] = 255

    return strong_edges, weak_edges, combined_edges

"""
Edge Linking

args:
    - image: (cv2 - gray image) image to apply Edge Linking to. Contains strong edges (255) and
                                weak edges (125)
return:
    - edges: (cv2 - gray image) image with edges
"""
def EdgeLinking(image):
    # Initialize edges
    edges = copy.deepcopy(image)

    # Loop through image
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            # If the pixel edge is weak
            if edges[i,j] == 125:
                # Check if the neighbors has a strong edge
                if edges[i+1][j] == 255 or edges[i-1][j] == 255 or edges[i][j+1] == 255 or edges[i][j-1] == 255 \
                    or edges[i+1][j+1] == 255 or edges[i-1][j-1] == 255 or edges[i+1][j-1] == 255 or edges[i-1][j+1] == 255:
                    edges[i,j] = 255 # Make weak edge strong
                else:
                    edges[i,j] = 0 # Delete weak edge

    return edges

if __name__ == '__main__':
    main()
