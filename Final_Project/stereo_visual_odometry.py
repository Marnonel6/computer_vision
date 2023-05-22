"""
Stereo visual odometry with the KITTI dataset

https://www.cvlibs.net/datasets/kitti/eval_odometry.php

Author: Marthinus (Marno) Nel
Date: 21 May 2023
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import pandas

class VisualOdometry():
    def __init__(self, dataset="07"):
        self.dataset = dataset
        self.dataset_path = "dataset"

        # Camera intrinsic parameters and Projection matrix
        self.P_l, self.P_r, self.K_l, self.K_r = self.import_calibration_parameters(self.dataset_path + "/sequences/" + self.dataset)
        print(f"P_l = {self.P_l}")
        print(f"P_r = {self.P_r}")
        print(f"K_l = {self.K_l}")
        print(f"K_r = {self.K_r}")

        # Ground truth poses
        self.GT_poses = self.import_ground_truth(self.dataset_path + "/poses/" + self.dataset + ".txt")
        # print(f"GT_poses = {self.GT_poses}")

        # Load stereo images into a list
        self.image_l, self.image_r = self.import_images(self.dataset_path + "/sequences/" + self.dataset)
        # print(f"image_l = {self.image_l}")
        # print(f"image_r = {self.image_r}")

    def import_images(self, image_dir_path):
        """
        Imports images into a list

        Parameters
        ----------
            image_dir_path (str): The relative path to the images directory

        Returns
        -------
            image_list_left (list): List of grayscale images
            image_list_right (list): List of grayscale images
        """
        image_l_path = image_dir_path + '/image_0'
        image_r_path = image_dir_path + '/image_1'

        image_l_path_list = [os.path.join(image_l_path, file) for file in sorted(os.listdir(image_l_path))]
        image_r_path_list = [os.path.join(image_r_path, file) for file in sorted(os.listdir(image_r_path))]

        image_list_left = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_l_path_list]
        image_list_right = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_r_path_list]

        return image_list_left, image_list_right

    def import_calibration_parameters(self, calibration_path):
        """
        Import camera intrinsic parameters and projection matrices

        Parameters
        ----------
            calibration_path (str): The relative path to the calibration file directory

        Returns
        -------
            P_l (np.array): Projection matrix for left camera
            P_r (np.array): Projection matrix for right camera
            K_l (np.array): Camera intrinsic parameters for left camera
            K_r (np.array): Camera intrinsic parameters for right camera
        """
        calib_file_path = calibration_path + '/calib.txt'
        calib_params = pandas.read_csv(calib_file_path, delimiter=' ', header=None, index_col=0)

        # Projection matrix
        P_l = np.array(calib_params.loc['P0:']).reshape((3,4))
        P_r = np.array(calib_params.loc['P1:']).reshape((3,4))
        # Camera intrinsic parameters
        K_l = cv2.decomposeProjectionMatrix(P_l)[0]
        K_r = cv2.decomposeProjectionMatrix(P_r)[0]

        return P_l, P_r, K_l, K_r

    def import_ground_truth(self, poses_path):
        """
        Import ground truth poses

        Parameters
        ----------
            poses_path (str): The relative path to the ground truth poses file directory

        Returns
        -------
            ground_truth (np.array): Ground truth poses
        """
        poses = pandas.read_csv(poses_path, delimiter=' ', header = None)
        ground_truth = poses.to_numpy()

        return ground_truth


def put_text(image, org, text, color=(0, 0, 255), fontScale=0.7, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    if not isinstance(org, tuple):
        (label_width, label_height), baseline = cv2.getTextSize(text, font, fontScale, thickness)
        org_w = 0
        org_h = 0

        h, w, *_ = image.shape

        place_h, place_w = org.split("_")

        if place_h == "top":
            org_h = label_height
        elif place_h == "bottom":
            org_h = h
        elif place_h == "center":
            org_h = h // 2 + label_height // 2

        if place_w == "left":
            org_w = 0
        elif place_w == "right":
            org_w = w - label_width
        elif place_w == "center":
            org_w = w // 2 - label_width // 2

        org = (org_w, org_h)

    image = cv2.putText(image, text, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    return image

def play_trip(l_frames, r_frames=None, lat_lon=None, timestamps=None, color_mode=False, playback_speed=10, win_name="Trip"):
    l_r_mode = r_frames is not None

    if not l_r_mode:
        r_frames = [None]*len(l_frames)

    frame_count = 0
    for i, frame_step in enumerate(zip(l_frames, r_frames)):
        img_l, img_r = frame_step

        if not color_mode:
            img_l = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)
            if img_r is not None:
                img_r = cv2.cvtColor(img_r, cv2.COLOR_GRAY2BGR)


        if img_r is not None:
            img_l = put_text(img_l, "top_center", "Left")
            img_r = put_text(img_r, "top_center", "Right")
            show_image = np.vstack([img_l, img_r])
        else:
            show_image = img_l
        show_image = put_text(show_image, "top_left", "Press ESC to stop")
        show_image = put_text(show_image, "top_right", f"Frame: {frame_count}/{len(l_frames)}")

        if timestamps is not None:
            time = timestamps[i]
            show_image = put_text(show_image, "bottom_right", f"{time}")


        if lat_lon is not None:
            lat, lon = lat_lon[i]
            show_image = put_text(show_image, "bottom_left", f"{lat}, {lon}")

        cv2.imshow(win_name, show_image)

        key = cv2.waitKey(playback_speed)
        if key == 27:  # ESC
            break
        frame_count += 1
    cv2.destroyWindow(win_name)

def main():
    """
    main function
    """
    SVO_dataset = VisualOdometry(dataset = "07")

    # Play images of the trip
    play_trip(SVO_dataset.image_l, SVO_dataset.image_r)

    # traj = visual_odometry(handler, detector='sift', matching='BF', filter_match_distance=0.45,
    #                         stereo_matcher='sgbm', mask=None)


if __name__ == "__main__":
    main()
