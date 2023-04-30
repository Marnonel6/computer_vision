"""
Template-matching based Target/Motion tracking

Author: Marthinus (Marno) Nel
Date: 04/30/2023
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from sklearn.cluster import KMeans 

def main():

    # NOTE Debug
    print("RUNNING!")

    # Load in images as BGR
    # test_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp6/test_images/test.bmp', cv2.IMREAD_COLOR)
    # Load all the images of the video
    video = []
    directory = '/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp7/test_images/video_images_girl/'
    img_files = os.listdir(directory)
    img_files.sort()
    for filename in img_files:
        if filename.endswith('.jpg') or filename.endswith('.png'):
            filepath = os.path.join(directory, filename)
            image = cv2.imread(filepath)
            # Get current dimensions of image
            height, width = image.shape[:2]
            # Double the size of the image
            resized_img = cv2.resize(image, (2*width, 2*height))
            video.append(resized_img)

    while True:
        for frame in video:
            cv2.imshow('Video', frame)
            if cv2.waitKey(100) & 0xFF == ord('q'): # Display each image for 50ms
                break
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    # NOTE Debug
    print("FINISHED!")

    # # Display images
    # Display_images = True
    # if Display_images:
    #     # Display images at each step
    #     plt.figure(1)
    #     plt.subplot(2, 3, 1)
    #     plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    #     plt.title('Input')
    #     plt.subplot(2,3,2)
    #     plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(test_img_magnitude), cv2.COLOR_GRAY2RGB))
    #     plt.title('Sobel Edge')
    #     plt.subplot(2,3,3)
    #     plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(test_img_hough), cv2.COLOR_GRAY2RGB))
    #     plt.gca().invert_yaxis()
    #     plt.xlabel('rho')
    #     plt.ylabel('theta')
    #     plt.title('Hough Transform')
    #     plt.subplot(2,3,4)
    #     plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(test_img_hough_filter), cv2.COLOR_GRAY2RGB))
    #     plt.gca().invert_yaxis()
    #     plt.xlabel('rho')
    #     plt.ylabel('theta')
    #     plt.title('Hough Transform Filtered')
    #     plt.subplot(2,3,5)
    #     plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(test_img_hough_filter), cv2.COLOR_GRAY2RGB))
    #     plt.scatter(test_centroids[:,0],test_centroids[:,1])
    #     plt.gca().invert_yaxis()
    #     plt.xlabel('rho')
    #     plt.ylabel('theta')
    #     plt.title('Cluster Centroids [K-means]')
    #     plt.subplot(2,3,6)
    #     plt.imshow(predicted_lines_test)
    #     plt.title('Predicted Lines')

    #     plt.figure(2)
    #     plt.subplot(2, 3, 1)
    #     plt.imshow(cv2.cvtColor(test2_img, cv2.COLOR_BGR2RGB))
    #     plt.title('Input')
    #     plt.subplot(2,3,2)
    #     plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(test2_img_magnitude), cv2.COLOR_GRAY2RGB))
    #     plt.title('Sobel Edge')
    #     plt.subplot(2,3,3)
    #     plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(test2_img_hough), cv2.COLOR_GRAY2RGB))
    #     plt.gca().invert_yaxis()
    #     plt.xlabel('rho')
    #     plt.ylabel('theta')
    #     plt.title('Hough Transform')
    #     plt.subplot(2,3,4)
    #     plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(test2_img_hough_filter), cv2.COLOR_GRAY2RGB))
    #     plt.gca().invert_yaxis()
    #     plt.xlabel('rho')
    #     plt.ylabel('theta')
    #     plt.title('Hough Transform Filtered')
    #     plt.subplot(2,3,5)
    #     plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(test2_img_hough_filter), cv2.COLOR_GRAY2RGB))
    #     plt.gca().invert_yaxis()
    #     plt.scatter(test2_centroids[:,0],test2_centroids[:,1])
    #     plt.xlabel('rho')
    #     plt.ylabel('theta')
    #     plt.title('Cluster Centroids [K-means]')
    #     plt.subplot(2,3,6)
    #     plt.imshow(predicted_lines_test2)
    #     plt.title('Predicted Lines')

    #     plt.figure(3)
    #     plt.subplot(2, 3, 1)
    #     plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    #     plt.title('Input')
    #     plt.subplot(2,3,2)
    #     plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(input_img_magnitude), cv2.COLOR_GRAY2RGB))
    #     plt.title('Sobel Edge')
    #     plt.subplot(2,3,3)
    #     plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(input_img_hough), cv2.COLOR_GRAY2RGB))
    #     plt.gca().invert_yaxis()
    #     plt.xlabel('rho')
    #     plt.ylabel('theta')
    #     plt.title('Hough Transform')
    #     plt.subplot(2,3,4)
    #     plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(input_img_hough_filter), cv2.COLOR_GRAY2RGB))
    #     plt.gca().invert_yaxis()
    #     plt.xlabel('rho')
    #     plt.ylabel('theta')
    #     plt.title('Hough Transform Filtered')
    #     plt.subplot(2,3,5)
    #     plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(input_img_hough_filter), cv2.COLOR_GRAY2RGB))
    #     plt.gca().invert_yaxis()
    #     plt.scatter(input_centroids[:,0],input_centroids[:,1])
    #     plt.xlabel('rho')
    #     plt.ylabel('theta')
    #     plt.title('Cluster Centroids [K-means]')
    #     plt.subplot(2,3,6)
    #     plt.imshow(cv2.cvtColor(predicted_lines_input, cv2.COLOR_BGR2RGB))
    #     plt.title('Predicted Lines')

    #     plt.show()

"""
<Function description>

args:
    - <Var name>: (<Type>) <Description>
return:
    - <Var name>: (<Type>) <Description>
"""

if __name__ == '__main__':
    main()
