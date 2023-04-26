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
    test_img_edge = Sobel(test_img, 100)
    test2_img_edge = Sobel(test2_img, 100)
    input_img_edge = Sobel(input_img, 50)

    # Display images at each step in Hough transform
    plt.figure(1)
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    plt.title('Input')
    plt.subplot(2,3,2)
    plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(test_img_edge), cv2.COLOR_GRAY2RGB))
    plt.title('Sobel Edge')
    # plt.subplot(2,3,3)
    # plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    # plt.title('Input')

    plt.figure(2)
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(test2_img, cv2.COLOR_BGR2RGB))
    plt.title('Test2 Image')
    plt.subplot(2,3,2)
    plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(test2_img_edge), cv2.COLOR_GRAY2RGB))
    plt.title('Sobel Edge')

    plt.figure(3)
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    plt.title('Input Image')
    plt.subplot(2,3,2)
    plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(input_img_edge), cv2.COLOR_GRAY2RGB))
    plt.title('Sobel Edge')

    plt.show()


"""
Edge detection with Sobel

args:
    - img: (cv2 - gray image) image to be processed
    - threshold: (int) threshold for edge detection [Lower threshold more edges and more noise]
return:
    - img: (cv2 - gray image) image with edges detected
"""
def Sobel(img, threshold):
    # Convert BGR to Gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply sobel filter to image
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

    # Normalize gradient magnitude
    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    # Threshold gradient magnitude
    gradient_magnitude[gradient_magnitude < threshold] = 0

    return gradient_magnitude



if __name__ == '__main__':
    main()
