import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():

    # Load in images as BGR
    gun1_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp5/test_images/gun1.bmp', cv2.IMREAD_COLOR)
    joy1_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp5/test_images/joy1.bmp', cv2.IMREAD_COLOR)
    pointer1_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp5/test_images/pointer1.bmp', cv2.IMREAD_COLOR)
    lena_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp5/test_images/lena.bmp', cv2.IMREAD_COLOR)
    test1_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp5/test_images/test1.bmp', cv2.IMREAD_COLOR)
    # Put images next to each other
    test_images = cv2.hconcat([np.uint8(gun1_img), np.uint8(joy1_img), np.uint8(pointer1_img)])

    # Display images grid
    result_images = cv2.hconcat([np.uint8(gun1_img), np.uint8(joy1_img), np.uint8(pointer1_img)])
    # Final images
    final_images = cv2.vconcat([test_images, result_images])
    final_images2 = cv2.hconcat([np.uint8(lena_img), np.uint8(GaussSmoothing(lena_img))])
    final_images3 = cv2.hconcat([np.uint8(test1_img), np.uint8(test1_img)])
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
return:
    - gs_smoothed_image: (cv2 - BGR image) gaussian smoothed image
"""
def GaussSmoothing(image, N=5, sigma=1):
    # Define the Gaussian filter kernel
    kernel = np.zeros((N, N), dtype=np.float32)
    mid = N // 2
    for i in range(-mid, mid+1):
        for j in range(-mid, mid+1):
            kernel[i+mid][j+mid] = np.exp(-(i*i + j*j) / (2 * sigma*sigma))
    kernel /= kernel.sum()

    # Convolve the kernel with the input image
    smoothed = cv2.filter2D(image, -1, kernel)

    """ Using OpenCV """
    # Convert to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # Apply gaussian smoothing with OpenCV
    # gs_smoothed_image = cv2.GaussianBlur(gray, (5, 5), 0)
    # smoothed = cv2.cvtColor(gs_smoothed_image, cv2.COLOR_GRAY2BGR)

    return smoothed



if __name__ == '__main__':
    main()