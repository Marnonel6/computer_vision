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

def main():
    """
    main function
    """
    SVO_dataset = VisualOdometry(dataset = "07")

    # Play images of the trip
    vs.play_trip(SVO_dataset.image_l, SVO_dataset.image_r)

    # traj = visual_odometry(handler, detector='sift', matching='BF', filter_match_distance=0.45,
    #                         stereo_matcher='sgbm', mask=None)

if __name__ == "__main__":
    main()
