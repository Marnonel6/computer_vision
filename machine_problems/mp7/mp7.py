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
    video_hsv = []
    directory = '/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp7/test_images/video_images_girl/'
    img_files = os.listdir(directory)
    img_files.sort()
    num_images = 0
    max_images = 10 # Maximum number of images to load
    for filename in img_files:
        if num_images <= max_images:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                filepath = os.path.join(directory, filename)
                image = cv2.imread(filepath, cv2.IMREAD_COLOR)
                # Convert BGR to HSV
                image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                # Get current dimensions of image
                height, width = image.shape[:2]
                # Double the size of the image
                resized_img = cv2.resize(image, (2*width, 2*height))
                resized_img_hsv = cv2.resize(image_HSV, (2*width, 2*height))
                # Add to video
                video.append(resized_img)
                video_hsv.append(resized_img_hsv)
                num_images += 1

    # Get the first frame
    first_frame = video[0]
    # Draw a red square on the image to indicate where the face is. The start face location is hard coded.
    # NOTE: (x1,y1) is the top left corner of the square
    # NOTE: (x2,y2) is the bottom right corner of the square
    # NOTE: (0,0) is the top left corner of the image
    # NOTE: (width,height) is the bottom right corner of the image
    bbox_size = [45, 50]    # Number of pixels x = 90, y = 100
    bbox_center = [145, 90]   # Fist image center of face bounding box
    # Define the top-left and bottom-right coordinates of the bounding box
    bbox = [bbox_center[0]-bbox_size[0], bbox_center[1]-bbox_size[1], bbox_center[0]+bbox_size[0], bbox_center[1]+bbox_size[1]]  # NOTE FACE START BOX IN FIRST FRAME
    bbox_color = (0, 0, 255) # BGR color for bounding box
    bbox_thickness = 2
    # Draw the bounding box on the image
    cv2.rectangle(video[0], (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, bbox_thickness)






    while True:
        for frame in video:
            cv2.imshow('Video', frame)
            if cv2.waitKey(5000) & 0xFF == ord('q'): # Display each image for 100ms
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
SSD - Sum of squared difference for motion tracking.
    - SSD = sum((I(x,y) - T(x,y))^2) over x and y in a predefined bounding box size
    - Minimize SSD for prediction: - T(x,y) is the target region in the previous frame
                                   - I(x,y) is the current frame's matching candidate
    - SSD is not robust to illumination changes, occlusion, large motions, large changes in scale, etc.

args:
    - video: (cv2 - BGR) BGR images in a list
    - video_hsv: (cv2 - HSV) HSV images in a list
    - bbox: bounding box of the target object in the first frame
    - search_window: search window size
    - step_size: step size for the search window
return:
    - motion_tracked_video: (cv2 - BGR) BGR images in a list with bounding box on tracked object
"""
# def motion_tracking(video, video_hsv, bbox, search_window, step_size):


if __name__ == '__main__':
    main()






    # # Initialize the bounding box
    # bbox = np.array(bbox)
    # bbox = bbox.astype(int)
    # bbox = bbox.reshape(1,4)
    # motion_tracked_video = []
    # for i in range(len(video)):
    #     # Get the current frame
    #     frame = video[i]
    #     frame_hsv = video_hsv[i]
    #     # Get the target region in the previous frame
    #     target_region = frame_hsv[bbox[0,1]:bbox[0,1]+bbox[0,3], bbox[0,0]:bbox[0,0]+bbox[0,2], 0]
    #     # Initialize the SSD
    #     ssd = np.inf
    #     # Initialize the candidate bounding box
    #     candidate_bbox = bbox

    #     # Search for the best candidate in the search window
    #     for y in range(bbox[0,1]-search_window, bbox[0,1]+search_window, step_size):
    #         for x in range(bbox[0,0]-search_window, bbox[0,0]+search_window, step_size):
    #             # Get the candidate region
    #             candidate_region = frame_hsv[y:y+bbox[0,3], x:x+bbox[0,2], 0]
    #             # Calculate the SSD
    #             ssd_candidate = np.sum(np.square(target_region - candidate_region))
    #             # Update the best candidate
    #             if ssd_candidate < ssd:
    #                 ssd = ssd_candidate
    #                 candidate_bbox = np.array([x, y, bbox[0,2], bbox[0,3]])
    #     # Update the bounding box
    #     bbox = candidate_bbox
    #     # Draw the bounding box
    #     cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,255,0), 2)
    #     # Append the frame to the motion tracked video
    #     motion_tracked_video.append(frame)
    # return motion_tracked_video