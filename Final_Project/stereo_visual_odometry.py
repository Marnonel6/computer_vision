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
import visualize as vs

class VisualOdometry():
    def __init__(self, dataset="07"):
        self.dataset = dataset
        self.dataset_path = "dataset"

        # Camera intrinsic parameters and Projection matrix
        self.P_l, self.P_r, self.K_l, self.K_r = self.import_calibration_parameters(self.dataset_path + "/sequences/" + self.dataset)
        # print(f"P_l = {self.P_l}")
        # print(f"P_r = {self.P_r}")
        # print(f"K_l = {self.K_l}")
        # print(f"K_r = {self.K_r}")

        # Ground truth poses
        self.GT_poses = self.import_ground_truth(self.dataset_path + "/poses/" + self.dataset + ".txt")
        # print(f"GT_poses = {self.GT_poses}")

        # Load stereo images into a list
        self.image_l_list, self.image_r_list = self.import_images(self.dataset_path + "/sequences/" + self.dataset)
        # print(f"image_l = {self.image_l_list}")
        # print(f"image_r = {self.image_r_list}")

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

    def feature_detection(self, detector, image):
        """
        Feature detection/extraction

        Parameters
        ----------
            detector (str): The type of feature detector to use
            image (np.array): The image to detect features in

        Returns
        -------
            keypoints (list): List of keypoints
            descriptors (list): List of descriptors
        """
        if detector == 'orb':
            orb = cv2.ORB_create()
            # Detects keypoints and computes corresponding feature descriptors and returns a list for each.
            keypoints, descriptors = orb.detectAndCompute(image, mask=None)
        elif detector == 'sift':
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(image, mask=None)
        # elif detector == 'surf':
        #     surf = cv2.SURF_create()
        #     keypoints, descriptors = surf.detectAndCompute(image, mask=None)
        else:
            raise Exception("Invalid detector type")

        return keypoints, descriptors
    
    def feature_matching(self, detector, descriptors_l_prev, descriptors_l_curr):
        """
        Feature matching

        Parameters
        ----------
            detector (str): The detector implies which matcher to use
            descriptors_l_prev (list): List of descriptors from previous (t-1) left image
            descriptors_l_curr (list): List of descriptors from current (t) left image

        Returns
        -------
            matches (list): List of matches
        """
        if detector == 'orb':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif detector == 'sift':
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        # elif detector == 'flann':
        #     # FLANN parameters
        #     FLANN_INDEX_LSH = 6
        #     index_params = dict(algorithm=FLANN_INDEX_LSH,
        #                         table_number=6,
        #                         key_size=12,
        #                         multi_probe_level=1)
        #     search_params = dict(checks=50)
        #     flann = cv2.FlannBasedMatcher(index_params, search_params)
        #     matches = flann.knnMatch(descriptors_l, descriptors_r, k=2)
        else:
            raise Exception("Invalid matcher type")

        matches = bf.match(descriptors_l_prev, descriptors_l_curr)

        return matches

def main():
    """
    main function
    """
    SVO_dataset = VisualOdometry(dataset = "07")

    # Choose feature detector type
    detector = "orb"

    # Play images of the trip
    vs.play_trip(SVO_dataset.image_l_list, SVO_dataset.image_r_list)

    """ Preform visual odometry on the dataset """

    # Initialize the trajectory
    estimated_traj = np.zeros((len(SVO_dataset.image_l_list), 3, 4))
    T_current = np.eye(4) # Start at identity matrix
    estimated_traj[0] = T_current[:3, :]

    # Setup visual odometry images
    image_l_curr = SVO_dataset.image_l_list[0]
    image_r_curr = SVO_dataset.image_r_list[0]

    # for i in range(len(SVO_dataset.image_l_list) - 1):
    for i in range(1):
        # Previous image
        image_l_prev = image_l_curr
        image_r_prev = image_r_curr
        # Current image
        image_l_curr = SVO_dataset.image_l_list[i + 1]
        image_r_curr = SVO_dataset.image_r_list[i + 1]

        # Feature detection/extraction
        keypoints_l_prev, descriptors_l_prev = SVO_dataset.feature_detection(detector, image_l_prev)
        keypoints_r_prev, descriptors_r_prev = SVO_dataset.feature_detection(detector, image_r_prev)
        keypoints_l_curr, descriptors_l_curr = SVO_dataset.feature_detection(detector, image_l_curr)
        keypoints_r_curr, descriptors_r_curr = SVO_dataset.feature_detection(detector, image_r_curr)
        # print(f"keypoints_l_prev = {keypoints_l_prev}")
        # print(f"descriptors_l_prev = {descriptors_l_prev}")

        # Feature matching
        matches_l = SVO_dataset.feature_matching(detector, descriptors_l_prev, descriptors_l_curr)
        # print(f"matches_l = {matches_l}")


if __name__ == "__main__":
    main()
