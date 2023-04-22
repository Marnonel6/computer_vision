import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import convolve

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
    smoothed_gun1_img = GaussSmoothing(gun1_img)
    smoothed_joy1_img = GaussSmoothing(joy1_img)
    smoothed_pointer1_img = GaussSmoothing(pointer1_img)
    smoothed_lena_img = GaussSmoothing(lena_img)
    smoothed_test1_img = GaussSmoothing(test1_img)
    # Smoothed images grid
    smoothed_images = cv2.hconcat([np.uint8(smoothed_gun1_img), np.uint8(smoothed_joy1_img), np.uint8(smoothed_pointer1_img)])

    # Compute image gradient
    mag_gun, dir_gun = ImageGradient(gun1_img)
    mag_joy, dir_joy = ImageGradient(joy1_img)
    mag_pointer, dir_pointer = ImageGradient(pointer1_img)
    mag_lena, dir_lena = ImageGradient(lena_img)
    mag_test, dir_test = ImageGradient(test1_img)

    # Final images
    final_images = cv2.vconcat([test_images, smoothed_images])
    final_images2 = cv2.hconcat([np.uint8(lena_img), np.uint8(smoothed_lena_img)])
    final_images3 = cv2.hconcat([np.uint8(test1_img), np.uint8(smoothed_test1_img)])
    # Display
    cv2.imshow('Canny edge detection', final_images)
    cv2.imshow('Canny edge detection2', final_images2)
    cv2.imshow('Canny edge detection3', final_images3)

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
    Ix = convolve(gray, Gx)
    Iy = convolve(gray, Gy)

    # Compute magnitude and direction
    magnitude = np.sqrt(Ix*Ix + Iy*Iy)
    direction = 180 + np.arctan2(Iy, Ix)*(180/np.pi) # In degrees


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

    return magnitude, direction


if __name__ == '__main__':
    main()