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
import math

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
    max_images = 500 # Maximum number of images to load
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

    # Draw a red square on the image to indicate where the face is. The start face location is hard coded.
    # NOTE: (x1,y1) is the top left corner of the square
    # NOTE: (x2,y2) is the bottom right corner of the square
    # NOTE: (0,0) is the top left corner of the image
    # NOTE: (width,height) is the bottom right corner of the image
    bbox_size = [45, 50]    # Number of pixels x = 90, y = 100
    bbox_center = [145, 90]   # Fist image center of face bounding box

    # Preform SSD - Sum of squared difference motion tracking
    object_tracked_video_SSD = motion_tracking_SSD(video, video_hsv, bbox_center, step_size=5) # NOTE step_size = 1 is the most accurate






    while True:
        for frame in object_tracked_video_SSD:
            cv2.imshow('object_tracked_video_SSD', frame)
            if cv2.waitKey(100) & 0xFF == ord('q'): # Display each image for 100ms
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
    - bbox_size: (list -> [x/2,y/2]) Size of the bounding box in number of pixels
    - bbox_center: (list -> [x,y]) Centre of object to track in first image
    - bbox_thickness: (int) Thickness of bounding box
    - bbox_color: ((B,G,R)) BGR color for bounding box
    - search_window: (int) search window size
    - step_size: (int) step size for the search window

return:
    - motion_tracked_video: (cv2 - BGR) BGR images in a list with bounding box on tracked object
"""
def motion_tracking_SSD(video, video_hsv, bbox_center, bbox_size=[45, 50], bbox_thickness=2, bbox_color=(0, 0, 255), search_window=30, step_size=1):
    # Deepcopy original video
    object_tracked_video_SSD = copy.deepcopy(video) # TODO makes slower so maybe not use

    # Define the top-left and bottom-right coordinates of the bounding box
    bbox = [bbox_center[0]-bbox_size[0], bbox_center[1]-bbox_size[1], bbox_center[0]+bbox_size[0],
            bbox_center[1]+bbox_size[1]]
    # Draw the bounding box NOTE FACE/OBJECT START BOX IN FIRST FRAME
    cv2.rectangle(object_tracked_video_SSD[0], (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                  bbox_color, bbox_thickness)

    # Loop through each frame in the video
    # for frame in range(1,500):
    for frame in range(1, len(video)):
        # Get the current and previous frame
        current_frame_hsv = video_hsv[frame]
        previous_frame_hsv = video_hsv[frame-1]
        # Get the target region in the previous frame
        target_region = previous_frame_hsv[bbox[1]:bbox[3], bbox[0]:bbox[2]] # [y1:y2, x1:x2]
        # Initialize the SSD
        min_SSD = math.inf
        # Initialize the candidate bounding box
        candidate_bbox = [0, 0, 0, 0]

        # Search for the best candidate in the search window
        # for y in range(bbox[1]-search_window, bbox[3]+search_window):
            # for x in range(bbox[0]-search_window, bbox[2]+search_window):
        # NOTE Full image exhaustive search
        # for y in range(bbox_size[1], current_frame_hsv.shape[0]-bbox_size[1], step_size):
        #     for x in range(bbox_size[0], current_frame_hsv.shape[1]-bbox_size[0], step_size):
        # NOTE Local search window exhaustive search
        # Calculate search space ensuring that the bounding box is inside the image
        if bbox[1]-search_window >= bbox_size[1]:
            y_min = bbox[1]-search_window
        else:
            y_min = bbox_size[1]
        if bbox[3]+search_window <= current_frame_hsv.shape[0]-bbox_size[1]:
            y_max = bbox[3]+search_window
        else:
            y_max = current_frame_hsv.shape[0]-bbox_size[1]
        
        if bbox[0]-search_window >= bbox_size[0]:
            x_min = bbox[0]-search_window
        else:
            x_min = bbox_size[0]
        if bbox[2]+search_window <= current_frame_hsv.shape[1]-bbox_size[0]:
            x_max = bbox[2]+search_window
        else:
            x_max = current_frame_hsv.shape[1]-bbox_size[0]


        # # Calculate y and x range for local search space
        # if bbox_center[1]-search_window >= bbox_size[1]:
        #     y_min = bbox_center[1]-search_window
        # else:
        #     y_min = bbox_size[1]+1
        # if bbox_center[1]+search_window <= current_frame_hsv.shape[0]-bbox_size[1]:
        #     y_max = bbox_center[1]+search_window
        # else:
        #     y_max = current_frame_hsv.shape[0]-bbox_size[1]
        
        # if bbox_center[0]-search_window >= bbox_size[0]:
        #     x_min = bbox_center[0]-search_window
        # else:
        #     x_min = bbox_size[0]
        # if bbox_center[0]+search_window <= current_frame_hsv.shape[1]-bbox_size[0]:
        #     x_max = bbox_center[0]+search_window
        # else:
        #     x_max = current_frame_hsv.shape[1]-bbox_size[0]

        # Loop through the local search space
        for y in range(y_min, y_max, step_size):
            for x in range(x_min, x_max, step_size):
                # Get the candidate region in the current frame
                candidate_region = current_frame_hsv[y-bbox_size[1]:y+bbox_size[1], x-bbox_size[0]:x+bbox_size[0]]
                # Calculate the SSD
                SSD = np.sum(np.square(candidate_region - target_region))
                # print(f"ssd = {SSD}")
                # Update the minimum SSD and candidate bounding box
                if SSD < min_SSD:
                    min_SSD = SSD
                    candidate_bbox = [x-bbox_size[0], y-bbox_size[1], x+bbox_size[0], y+bbox_size[1]]

        # Update the bounding box
        bbox = candidate_bbox
        bbox_center = [bbox[0]-bbox[2],bbox[1]-bbox[3]]
        # Draw the bounding box
        cv2.rectangle(object_tracked_video_SSD[frame], (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      bbox_color, bbox_thickness)


    return object_tracked_video_SSD

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