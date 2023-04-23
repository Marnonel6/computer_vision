import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import convolve
from scipy.signal import convolve2d

def main():

    # Load in images as BGR
    gun1_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp5/test_images/gun1.bmp', cv2.IMREAD_COLOR)
    joy1_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp5/test_images/joy1.bmp', cv2.IMREAD_COLOR)
    pointer1_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp5/test_images/pointer1.bmp', cv2.IMREAD_COLOR)
    lena_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp5/test_images/lena.bmp', cv2.IMREAD_COLOR)
    test1_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp5/test_images/test1.bmp', cv2.IMREAD_COLOR)
    # Put images next to each other
    test_images = cv2.hconcat([np.uint8(gun1_img), np.uint8(joy1_img), np.uint8(pointer1_img)])

    # Preform gaussian smoothing
    # smoothed_gun1_img = GaussSmoothing(gun1_img)
    # smoothed_joy1_img = GaussSmoothing(joy1_img)
    # smoothed_pointer1_img = GaussSmoothing(pointer1_img)
    smoothed_lena_img = GaussSmoothing(lena_img)
    smoothed_test1_img = GaussSmoothing(test1_img)
    # Smoothed images grid
    # smoothed_images = cv2.hconcat([np.uint8(smoothed_gun1_img), np.uint8(smoothed_joy1_img), np.uint8(smoothed_pointer1_img)])


    # Gaussian blur with opencv #NOTE OPENCV TEST
    # opencv_test_image = cv2.GaussianBlur(test1_img, (5,5), 1.4)
    # cv2.imshow('opencv_test_image', opencv_test_image)
    # # Compute gradient and magnitude with openvc
    # opencv_mag_test, opencv_dir_test = cv2.cartToPolar(cv2.Sobel(smoothed_test1_img, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(smoothed_test1_img, cv2.CV_32F, 0, 1, ksize=3))
    # cv2.imshow('opencv_mag_test', opencv_mag_test)
    # cv2.imshow('opencv_dir_test', opencv_dir_test)
    # Compute nonmaxima suppression with opencv
    # opencv_nms_test = cv2.Canny(pointer1_img, 25, 150)
    # cv2.imshow('opencv_nms_test', opencv_nms_test)

    # Compute image gradient
    # mag_gun, dir_gun = ImageGradient(smoothed_gun1_img)
    # mag_joy, dir_joy = ImageGradient(smoothed_joy1_img)
    # mag_pointer, dir_pointer = ImageGradient(smoothed_pointer1_img)
    mag_lena, dir_lena = ImageGradient(smoothed_lena_img)
    mag_test, dir_test = ImageGradient(smoothed_test1_img)


    cv2.imshow('mag_lena', mag_lena.astype(np.int8))
    cv2.imshow('dir_lena', dir_lena.astype(np.int8))
    cv2.imshow('dir_test', dir_test.astype(np.int8)) # TODO ADD BACK
    cv2.imshow('mag_test', mag_test.astype(np.int8)) # TODO ADD BACK
    # cv2.imshow('Gradient3', dir_pointer)

    # Compute thresholds
    # T_high_gun, T_low_gun = compute_thresholds(mag_gun)
    # T_high_joy, T_low_joy = compute_thresholds(mag_joy)
    # T_high_pointer, T_low_pointer = compute_thresholds(mag_pointer)
    T_high_lena, T_low_lena = compute_thresholds(mag_lena)
    T_high_test, T_low_test = compute_thresholds(mag_test)

    # Apply non-maximum suppression
    # nms_gun = NonMaxSuppression(mag_gun, dir_gun)
    # nms_joy = NonMaxSuppression(mag_joy, dir_joy)
    # nms_pointer = NonMaxSuppression(mag_pointer, dir_pointer)
    nms_lena = NonMaxSuppression(mag_lena, dir_lena)
    nms_test = NonMaxSuppression(mag_test, dir_test)

    cv2.imshow('NonMaxSuppression lena', np.uint8(cv2.cvtColor(cv2.convertScaleAbs(NonMaxSuppression(mag_lena, dir_lena)), cv2.COLOR_GRAY2BGR)))
    cv2.imshow('NonMaxSuppression test', np.uint8(cv2.cvtColor(cv2.convertScaleAbs(NonMaxSuppression(mag_test, dir_test)), cv2.COLOR_GRAY2BGR)))

    # Display non-maxima suppresion
    # cv2.imshow('Non-maxima suppression', cv2.hconcat([np.uint8(nms_gun), np.uint8(nms_joy), np.uint8(nms_pointer)]))
    # cv2.imshow('Non-maxima suppression2', np.uint8(nms_lena))
    # cv2.imshow('Non-maxima suppression3', np.uint8(nms_test))

    # Final images
    # final_images = cv2.vconcat([test_images, smoothed_images])
    # final_images2 = cv2.hconcat([np.uint8(lena_img), np.uint8(smoothed_lena_img),
    #                              np.uint8(cv2.cvtColor(cv2.convertScaleAbs(nms_lena), cv2.COLOR_GRAY2BGR))])
    # final_images3 = cv2.hconcat([np.uint8(test1_img), np.uint8(smoothed_test1_img),
    #                             #  np.uint8(cv2.cvtColor(cv2.convertScaleAbs(mag_test), cv2.COLOR_GRAY2BGR)),
    #                             #  np.uint8(cv2.cvtColor(cv2.convertScaleAbs(dir_test*100), cv2.COLOR_GRAY2BGR)),
    #                              np.uint8(cv2.cvtColor(cv2.convertScaleAbs(nms_test), cv2.COLOR_GRAY2BGR))]) # HACK * 100 to see directions
    # # Display
    # cv2.imshow('Canny edge detection', final_images)
    # cv2.imshow('Canny edge detection2', final_images2)
    # cv2.imshow('Canny edge detection3', final_images3)

    # Wait for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
Gaussian smoothing

args:
    - image: (cv2 - BGR image) image to apply gaussian smoothing to
    - N: (int) size of the gaussian kernel [NxN]
    - sigma: (float) standard deviation of the gaussian distribution
return:
    - smoothed: (cv2 - BGR image) gaussian smoothed image
"""
def GaussSmoothing(image, N=5, sigma=1):
    # Define the Gaussian filter kernel
    # kernel = np.zeros((N, N), dtype=np.float32)
    # mid = N // 2
    # for i in range(-mid, mid+1):
    #     for j in range(-mid, mid+1):
    #         kernel[i+mid][j+mid] = np.exp(-(i*i + j*j) / (2 * sigma*sigma))
    # kernel /= kernel.sum()

    size = int(N) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    kernel =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

    # Convolve the kernel with the input image
    smoothed = cv2.filter2D(image, -1, kernel)

    """ Using OpenCV """
    # Convert to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # Apply gaussian smoothing with OpenCV
    # gs_smoothed_image = cv2.GaussianBlur(gray, (5, 5), 0)
    # smoothed = cv2.cvtColor(gs_smoothed_image, cv2.COLOR_GRAY2BGR)

    return smoothed

"""
Image gradient [Magnitude, Direction]

args:
    - image: (cv2 - BGR image) image to apply image gradient to
return:
    - magnitude: (cv2 - BGR image) magnitude of the image gradient
    - direction: (cv2 - BGR image) direction of the image gradient
"""
def ImageGradient(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sobel operators
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Apply Sobel convolution kernels
    # Ix = convolve(gray, Gx)
    # Iy = convolve(gray, Gy)
    Ix = convolve2d(Gx, gray)
    Iy = convolve2d(Gy, gray)

    # Compute magnitude and direction
    magnitude = np.sqrt(Ix*Ix + Iy*Iy)
    magnitude = (magnitude/magnitude.max()) * 255
    direction = np.arctan2(Iy, Ix) #*(180/np.pi) + 180# In degrees

    # magnitude, direction = cv2.cartToPolar(Ix, Iy)


    # # Convert to grayscale # NOTE OpenCV implimentation
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # Apply sobel operator
    # sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    # sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    # # Compute magnitude and direction
    # magnitude = np.sqrt(sobelx*sobelx + sobely*sobely)
    # direction = np.arctan2(sobely, sobelx)
    # # Convert to BGR
    # magnitude = cv2.cvtColor(np.uint8(magnitude), cv2.COLOR_GRAY2BGR)
    # direction = cv2.cvtColor(np.uint8(direction), cv2.COLOR_GRAY2BGR)

    # return magnitude.astype(np.int8), direction.astype(np.int8) #TODO ADD BACK
    return magnitude, direction

"""
Determining high and low thresholds

args:
    - image: (cv2 - BGR image) image to apply thresholds to
    - ratio: (float) ratio of high to low threshold
    - percentageOfNonEdge: (float) percentage of non-edge pixels
return:
    - high_threshold: (float) high threshold value
    - low_threshold: (float) low threshold value
"""
def compute_thresholds(image, ratio=0.5, percentageOfNonEdge=0.8):

    # Histogram intensity data from image
    hist, bins = np.histogram(image, bins=256, range=(0, 256))
    # Calculate cumulative distribution
    cumulative_hist = np.cumsum(hist)
    # Normalize the cumulative values
    cumulative_normalized = cumulative_hist / np.max(cumulative_hist)

    # NOTE Debug
    # # Plot cumulative distribution
    # plt.figure()
    # plt.plot(bins[:-1], cumulative_normalized)
    # # Set the axis labels and title
    # plt.xlabel('Input image pixel intensity')
    # plt.ylabel('Output image pixel intensity normalized')
    # plt.title('Cumulative histogram distribution')
    # plt.show()

    # Compute high and low thresholds
    high_threshold = np.argwhere(cumulative_normalized >= percentageOfNonEdge)[0][0]
    # print(high_threshold)
    low_threshold = high_threshold * ratio

    return high_threshold, low_threshold

"""
Non-maximum suppression

args:
    - image: (cv2 - BGR image) image to apply non-maximum suppression to
    - direction: (numpy array) direction of the image gradient
    - magnitude: (numpy array) magnitude of the image gradient
return:
    - suppressed: (cv2 - BGR image) non-maximum suppressed image
"""
# NOTE img -> mag
def NonMaxSuppression(mag, angle):
    # Convert to grayscale
    # mag = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row, col = mag.shape
    # Z = np.zeros((row, col), dtype=np.int8)
    # Suppressed image
    suppressed = np.zeros(mag.shape)
    angle = angle * 180 / np.pi
    angle[angle < 0] += 180

    # Loop through the image and find local maxima
    for i in range(1, col-1):
        for j in range(1, row-1):
            # try:
                # Initialize dx and dj as max value
                dx = 255
                dj = 255
                
               # Constrain to matrix angle = 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    dx, dj = mag[i, j+1], mag[i, j-1]
                # Constrain to matrix angle =  45
                elif (22.5 <= angle[i,j] < 67.5):
                    dx, dj = mag[i+1, j-1], mag[i-1, j+1]
                # Constrain to matrix angle =  90
                elif (67.5 <= angle[i,j] < 112.5):
                    dx, dj = mag[i+1, j], mag[i-1, j]
                # Constrain to matrix angle =  135
                elif (112.5 <= angle[i,j] < 157.5):
                    dx, dj = mag[i-1, j-1], mag[i+1, j+1]

                # Only keep pixel value of two sides are less intense than the current pixel
                if (mag[i,j] >= dx) and (mag[i,j] >= dj):
                    suppressed[i,j] = mag[i,j]
                else:
                    suppressed[i,j] = 0

            # except IndexError as e:
            #     pass

    return suppressed

if __name__ == '__main__':
    main()
