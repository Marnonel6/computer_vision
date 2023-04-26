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

    # Hough transform
    test_img_hough = HoughTransform(test_img_edge, 0.5)
    test2_img_hough = HoughTransform(test2_img_edge, 0.5)
    input_img_hough = HoughTransform(input_img_edge, 0.5)

    # Display images at each step in Hough transform
    plt.figure(1)
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    plt.title('Input')
    plt.subplot(2,3,2)
    plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(test_img_edge), cv2.COLOR_GRAY2RGB))
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
    plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(test2_img_edge), cv2.COLOR_GRAY2RGB))
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
    plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(input_img_edge), cv2.COLOR_GRAY2RGB))
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
    precision_scale = 2

    # Initialize the maximum size of the polar space as the range min to max of theta and rho
    polar_space_voting = np.zeros((int(max_theta*2*ratio*precision_scale), int(max_rho*2*precision_scale)))

    # Loop through image and vote for lines
    for y in range(row):
        for x in range(col):
            if img[y, x] > threshold:
                # for theta in range(int(min_theta), int(max_theta)):
                for theta in np.arange(min_theta, max_theta-0.1, 0.01):
                    rho = x * np.cos(theta) + y * np.sin(theta)
                    polar_space_voting[int((theta + max_theta)*ratio*precision_scale), int((rho + max_rho)*precision_scale)] += 1 # NOTE to make axis positive and not to -pi/2

    # NOTE Debug
    print("Done!")

    # Scale polar_space_voting to have values between 50 and 255
    polar_space_voting *= 150.0 / polar_space_voting.max()
    # polar_space_voting += 50
    # # Add 50 if pixel value does not equal 0
    polar_space_voting[polar_space_voting != 0] += 150


    # polar_space_voting *= 255.0 / polar_space_voting.max()

    return polar_space_voting




if __name__ == '__main__':
    main()
