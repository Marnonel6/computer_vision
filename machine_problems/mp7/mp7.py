"""
Template-matching based Target/Motion tracking
    - SSD: Sum of Squared Difference
    - CC: Cross-Correlation
    - NCC: Normalized Cross-Correlation

Author: Marthinus (Marno) Nel
Date: 04/30/2023
"""

import cv2
import numpy as np
import os
import copy
import math
import time

def main():

    # NOTE Debug
    print("RUNNING!")

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
    # bbox_size = [45, 50]    # Number of pixels x = 90, y = 100
    # bbox_center = [145, 90]   # Fist image center of face bounding box
    # NOTE or use GUI to select ROI - Region of interest and then press enter
    ROI = cv2.selectROI(video[0], fromCenter=False) # returns [x1, y1, width, height] - Left top (x1,y1)
    bbox_size = [int(ROI[2]/2), int(ROI[3]/2)]                     # Number of pixels [x, y]
    bbox_center = [ROI[0] + int(ROI[2]/2), ROI[1] + int(ROI[3]/2)] # Fist image center of face bounding box

    """ Preform each motion tracking algorithm separately """
    # Preform SSD - Sum of squared difference motion tracking (RED)
    # NOTE step_size = 1 & search_window = 10 is the most accurate and time efficient
    object_tracked_video_SSD = motion_tracking_SSD(video, video_hsv, bbox_center, bbox_size,
                                                   search_window=10, step_size=2, bbox_color=(0, 0, 255))

    # Preform CC - Cross correlation motion tracking (BLUE)
    # NOTE step_size = 1 & search_window = 10 is the most accurate and time efficient
    object_tracked_video_CC = motion_tracking_Cross_correlation(video, video_hsv, bbox_center,
                                                                bbox_size, search_window=10, step_size=2,
                                                                bbox_color=(255, 0, 0))

    # Preform NCC - Normalized Cross-correlation motion tracking (GREEN)
    # NOTE step_size = 1 & search_window = 10 is the most accurate and time efficient
    object_tracked_video_NCC = motion_tracking_NCC(video, video_hsv, bbox_center, bbox_size,
                                                   search_window=10, step_size=5, bbox_color=(0, 255, 0))

    """ Display all motion tracking algorithms in one video """
        # Preform SSD - Sum of squared difference motion tracking (RED)
    # NOTE step_size = 1 & search_window = 10 is the most accurate and time efficient
    object_tracked_video_SSD_CC_NCC = motion_tracking_SSD(object_tracked_video_NCC, video_hsv, bbox_center, bbox_size,
                                                   search_window=10, step_size=2, bbox_color=(0, 0, 255))

    # Preform CC - Cross correlation motion tracking (BLUE)
    # NOTE step_size = 1 & search_window = 10 is the most accurate and time efficient
    object_tracked_video_SSD_CC_NCC = motion_tracking_Cross_correlation(object_tracked_video_SSD_CC_NCC, video_hsv, bbox_center,
                                                                bbox_size, search_window=10, step_size=2,
                                                                bbox_color=(255, 0, 0))

    # Save result videos:
    save_result_videos = True
    if save_result_videos:
        fps = 10

        # VideoWriter object
        Video_obj = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('object_tracked_video_SSD.mp4', Video_obj, fps, (object_tracked_video_SSD[0].shape[1], object_tracked_video_SSD[0].shape[0]))  # Filename, Codec, FPS, Frame size
        # Iterate over the photos
        for frame in object_tracked_video_SSD:
            out.write(frame)
        # Release the VideoWriter object
        out.release()

        # VideoWriter object
        Video_obj = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('object_tracked_video_CC.mp4', Video_obj, fps,
                              (object_tracked_video_CC[0].shape[1], object_tracked_video_CC[0].shape[0]))  # Filename, Codec, FPS, Frame size
        # Iterate over the photos
        for frame in object_tracked_video_CC:
            out.write(frame)
        # Release the VideoWriter object
        out.release()

        # VideoWriter object
        Video_obj = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('object_tracked_video_NCC.mp4', Video_obj, fps,
                              (object_tracked_video_NCC[0].shape[1], object_tracked_video_NCC[0].shape[0]))  # Filename, Codec, FPS, Frame size
        # Iterate over the photos
        for frame in object_tracked_video_NCC:
            out.write(frame)
        # Release the VideoWriter object
        out.release()

        # VideoWriter object
        Video_obj = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('object_tracked_video_SSD_CC_NCC.mp4', Video_obj, fps,
                              (object_tracked_video_SSD_CC_NCC[0].shape[1], object_tracked_video_SSD_CC_NCC[0].shape[0]))  # Filename, Codec, FPS, Frame size
        # Iterate over the photos
        for frame in object_tracked_video_SSD_CC_NCC:
            out.write(frame)
        # Release the VideoWriter object
        out.release()

    # NOTE Debug
    print("FINISHED!")

    # Display the video
    while True:
        for frame in object_tracked_video_SSD_CC_NCC:
            cv2.imshow('object_tracked_video_SSD_CC_NCC', frame)
            if cv2.waitKey(100) & 0xFF == ord('q'): # Display each image for 100ms
                break
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

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
    - motion_tracked_video_SSD: (cv2 - BGR) BGR images in a list with bounding box on tracked object
"""
def motion_tracking_SSD(video, video_hsv, bbox_center, bbox_size=[45, 50], bbox_thickness=2,
                        bbox_color=(0, 0, 255), search_window=25, step_size=1):
    # Algorithm start time
    start_time = time.time()

    print("\n")

    # Deepcopy original video
    object_tracked_video_SSD = copy.deepcopy(video) # TODO makes slower so maybe not use

    # Define the top-left and bottom-right coordinates of the bounding box
    bbox = [bbox_center[0]-bbox_size[0], bbox_center[1]-bbox_size[1], bbox_center[0]+bbox_size[0],
            bbox_center[1]+bbox_size[1]]
    # Draw the bounding box NOTE FACE/OBJECT START BOX IN FIRST FRAME
    cv2.rectangle(object_tracked_video_SSD[0], (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                  bbox_color, bbox_thickness)

    # Loop through each frame in the video
    for frame in range(1, len(video)):
        # NOTE Algorithm Status
        time_passed = time.time() - start_time
        print(f"\r Sum of squared difference: Frame [{frame+1}/500] Time:{time.time() - start_time:.2f}/{(((500 - frame+1)/frame+1) * time_passed):.2f} s", end="")
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

        # Loop through the local search space
        for y in range(y_min, y_max, step_size):
            for x in range(x_min, x_max, step_size):
                # Get the candidate region in the current frame
                candidate_region = current_frame_hsv[y-bbox_size[1]:y+bbox_size[1], x-bbox_size[0]:x+bbox_size[0]]
                # Calculate the SSD
                SSD = np.sum(pow(candidate_region - target_region,2))
                # Update the minimum SSD and candidate bounding box
                if SSD < min_SSD:
                    min_SSD = SSD
                    candidate_bbox = [x-bbox_size[0], y-bbox_size[1], x+bbox_size[0], y+bbox_size[1]]

        # Update the bounding box
        bbox = candidate_bbox
        bbox_center = [bbox[0]-bbox[2], bbox[1]-bbox[3]]
        # Draw the bounding box
        cv2.rectangle(object_tracked_video_SSD[frame], (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      bbox_color, bbox_thickness)

    print(f"\n")

    return object_tracked_video_SSD

"""
CC  - Cross-Correlation for motion tracking.
    - CC = sum(I(x,y)*T(x,y)) over x and y in a predefined bounding box size
    - Maximize CC for prediction: - T(x,y) is the target region in the previous frame
                                  - I(x,y) is the current frame's matching candidate

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
    - motion_tracked_video_CC: (cv2 - BGR) BGR images in a list with bounding box on tracked object
"""
def motion_tracking_Cross_correlation(video, video_hsv, bbox_center, bbox_size=[45, 50], 
                                      bbox_thickness=2, bbox_color=(0, 0, 255), search_window=25, step_size=1):
    # Algorithm start time
    start_time = time.time()

    # Deepcopy original video
    object_tracked_video_CC = copy.deepcopy(video) # TODO makes slower so maybe not use

    # Define the top-left and bottom-right coordinates of the bounding box
    bbox = [bbox_center[0]-bbox_size[0], bbox_center[1]-bbox_size[1], bbox_center[0]+bbox_size[0],
            bbox_center[1]+bbox_size[1]]
    # Draw the bounding box NOTE FACE/OBJECT START BOX IN FIRST FRAME
    cv2.rectangle(object_tracked_video_CC[0], (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                  bbox_color, bbox_thickness)

    # Loop through each frame in the video
    for frame in range(1, len(video)):
        # NOTE Algorithm Status
        time_passed = time.time() - start_time
        print(f"\r Cross-Correlation: Frame [{frame+1}/500] Time:{time.time() - start_time:.2f}/{(((500 - frame+1)/frame+1) * time_passed):.2f} s", end="")

        # Get the current and previous frame
        current_frame_hsv = video_hsv[frame]
        previous_frame_hsv = video_hsv[frame-1]
        # Get the target region in the previous frame
        target_region = previous_frame_hsv[bbox[1]:bbox[3], bbox[0]:bbox[2]] # [y1:y2, x1:x2]
        # Initialize the CC (Cross-correlation) value 
        max_CC = -math.inf
        # Initialize the candidate bounding box
        candidate_bbox = [0, 0, 0, 0]

        # Search for the best candidate in the search window
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

        # Loop through the local search space
        for y in range(y_min, y_max, step_size):
            for x in range(x_min, x_max, step_size):
                # Get the candidate region in the current frame
                candidate_region = current_frame_hsv[y-bbox_size[1]:y+bbox_size[1], x-bbox_size[0]:x+bbox_size[0]]
                # Calculate the CC
                CC = np.sum(candidate_region*target_region)
                # Update the maximum CC (Cross-correlation) value and candidate bounding box
                if CC > max_CC:
                    max_CC = CC
                    candidate_bbox = [x-bbox_size[0], y-bbox_size[1], x+bbox_size[0], y+bbox_size[1]]

        # Update the bounding box
        bbox = candidate_bbox
        bbox_center = [bbox[0]-bbox[2], bbox[1]-bbox[3]]
        # Draw the bounding box
        cv2.rectangle(object_tracked_video_CC[frame], (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      bbox_color, bbox_thickness)

    print(f"\n")

    return object_tracked_video_CC

"""
NCC  - Normalized Cross-Correlation for motion tracking.
     - Maximize NCC for prediction: - T(x,y) is the target region in the previous frame
                                    - I(x,y) is the current frame's matching candidate
     - NCC = A / B over x and y in a predefined bounding box size
        - A = sum((I(x,y) - I_avg) * (T(x,y) - T_avg))
        - B = sqrt(sum((I(x,y) - I_avg)^2) * sum((T(x,y) - T_avg)^2))
        - I_avg = mean(I(x,y), axis=(0, 1))
        - T_avg = mean(T(x,y), axis=(0, 1))

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
    - motion_tracked_video_NCC: (cv2 - BGR) BGR images in a list with bounding box on tracked object
"""
def motion_tracking_NCC(video, video_hsv, bbox_center, bbox_size=[45, 50], bbox_thickness=2,
                        bbox_color=(0, 0, 255), search_window=25, step_size=1):
    # Algorithm start time
    start_time = time.time()

    # Deepcopy original video
    object_tracked_video_NCC = copy.deepcopy(video) # TODO makes slower so maybe not use

    # Define the top-left and bottom-right coordinates of the bounding box
    bbox = [bbox_center[0]-bbox_size[0], bbox_center[1]-bbox_size[1], bbox_center[0]+bbox_size[0],
            bbox_center[1]+bbox_size[1]]
    # Draw the bounding box NOTE FACE/OBJECT START BOX IN FIRST FRAME
    cv2.rectangle(object_tracked_video_NCC[0], (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                  bbox_color, bbox_thickness)

    # Loop through each frame in the video
    for frame in range(1, len(video)):
        # NOTE Algorithm Status
        time_passed = time.time() - start_time
        print(f"\r Normalized Cross-Correlation: Frame [{frame+1}/500] Time:{time.time() - start_time:.2f}/{(((500 - frame+1)/frame+1) * time_passed):.2f} s", end="")

        # Get the current and previous frame
        current_frame_hsv = video_hsv[frame]
        previous_frame_hsv = video_hsv[frame-1]
        # Get the target region in the previous frame
        target_region = previous_frame_hsv[bbox[1]:bbox[3], bbox[0]:bbox[2]] # [y1:y2, x1:x2]
        # Initialize the NCC
        max_NCC = -math.inf
        # Initialize the candidate bounding box
        candidate_bbox = [0, 0, 0, 0]

        # Search for the best candidate in the search window
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

        # Loop through the local search space
        for y in range(y_min, y_max, step_size):
            for x in range(x_min, x_max, step_size):
                # Get the candidate region in the current frame
                candidate_region = current_frame_hsv[y-bbox_size[1]:y+bbox_size[1], x-bbox_size[0]:x+bbox_size[0]]
                # NOTE: Calculate the NCC
                # Calculate the average value of the 3 axis in candidate_region
                I_avg = np.mean(candidate_region, axis=(0, 1))
                # Calculate the average value of the 3 axis in target_region
                T_avg = np.mean(target_region, axis=(0, 1))
                # Calculate NCC
                numerator = np.sum((candidate_region-I_avg)*(target_region-T_avg))
                denominator = np.sqrt(np.sum(np.square(candidate_region - I_avg))*np.sum(np.square(target_region - T_avg)))
                NCC = numerator/denominator
                # Update the minimum NCC and candidate bounding box
                if NCC > max_NCC:
                    max_NCC = NCC
                    candidate_bbox = [x-bbox_size[0], y-bbox_size[1], x+bbox_size[0], y+bbox_size[1]]

        # Update the bounding box
        bbox = candidate_bbox
        bbox_center = [bbox[0]-bbox[2], bbox[1]-bbox[3]]
        # Draw the bounding box
        cv2.rectangle(object_tracked_video_NCC[frame], (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      bbox_color, bbox_thickness)

    print(f"\n")

    return object_tracked_video_NCC

if __name__ == '__main__':
    main()
