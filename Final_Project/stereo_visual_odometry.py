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
        self.P_l, self.P_r, self.K_l, self.K_r, self.t_l, self.t_r = self.import_calibration_parameters(self.dataset_path + "/sequences/" + self.dataset)
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
        K_l, R_l, t_l, _, _, _, _  = cv2.decomposeProjectionMatrix(P_l)
        K_r, R_r, t_r, _, _, _, _ = cv2.decomposeProjectionMatrix(P_r)

        # Normalize translation vectors to non-homogenous (euclidean) coordinates
        t_l = (t_l / t_l[3])[:3]
        t_r = (t_r / t_r[3])[:3]

        return P_l, P_r, K_l, K_r, t_l, t_r

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
        # ground_truth = poses.to_numpy()

        ground_truth = np.zeros((len(poses),3,4))
        for i in range(len(poses)):
            ground_truth[i] = np.array(poses.iloc[i]).reshape((3,4))

        return ground_truth

    def feature_detection(self, detector, image, mask=None): # TODO do mask!!
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
            detect = cv2.ORB_create()
            # Detects keypoints and computes corresponding feature descriptors and returns a list for each.
        elif detector == 'sift':
            detect = cv2.SIFT_create()
        # elif detector == 'surf':
        #     surf = cv2.SURF_create()
        #     keypoints, descriptors = surf.detectAndCompute(image, mask)
        else:
            raise Exception("Invalid detector type")

        keypoints, descriptors = detect.detectAndCompute(image, mask)

        return keypoints, descriptors
    
    def feature_matching(self, matcher, detector, descriptors_l_prev, descriptors_l_curr, knn = True, k = 2, sort = False):
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
        if matcher == 'bf':
            if detector == 'orb': # TODO try HAMMING2
                match = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) # TODO Try True without ratio test
            elif detector == 'sift':
                match = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        elif detector == 'flann':
            # FLANN parameters
            # FLANN_INDEX_LSH = 6
            # index_params = dict(algorithm=FLANN_INDEX_LSH,
            #                     table_number=6,
            #                     key_size=12,
            #                     multi_probe_level=1)
            # search_params = dict(checks=50)
            # flann = cv2.FlannBasedMatcher(index_params, search_params)
            # matches = flann.knnMatch(descriptors_l_prev, descriptors_l_curr, k)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            match = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise Exception("Invalid matcher type")

        # Match features
        if knn == True:
            matches = match.knnMatch(descriptors_l_prev, descriptors_l_curr, k)
        else:
            matches = match.match(descriptors_l_prev, descriptors_l_curr)

        if sort == True:
            matches = sorted(matches, key = lambda x:x[0].distance)

        return matches

    def create_mask(self, depth_map):
        """
        Create mask for feature detection

        Parameters
        ----------
            depth_map (np.array): The depth map

        Returns
        -------
            mask (np.array): The mask
        """
        mask = np.zeros(depth_map.shape, dtype = np.uint8)
        y_max = depth_map.shape[0]
        x_max = depth_map.shape[1]
        cv2.rectangle(mask, (96, 0), (x_max, y_max), (255), thickness = -1)
        # plt.imshow(mask)
        return mask

    def feature_match_visualize(self, keypoints_l_prev, keypoints_l_curr, image_l_prev, image_l_curr, matches):
        """
        Visualize feature matches

        Parameters
        ----------
            keypoints_l_prev (list): List of keypoints from previous (t-1) left image
            keypoints_l_curr (list): List of keypoints from current (t) left image
            image_l_prev (np.array): Previous (t-1) left image
            image_l_curr (np.array): Current (t) left image
            matches (list): List of matches

        Returns
        -------
            matches_image (np.array): Image with feature matches drawn
        """
        # Draw matches
        matches_image = cv2.drawMatches(image_l_prev, keypoints_l_prev, image_l_curr, keypoints_l_curr, matches, None, flags=2)
        plt.figure(figsize = (16,6), dpi = 100)
        plt.imshow(matches_image)

    def compute_disparity_map(self, image_l, image_r, matcher):
        """
        Compute disparity map:
            ParametersA disparity map is a visual representation that shows the pixel-wise
            horizontal shift or difference between corresponding points in a pair of stereo images,
            providing depth information for the scene.

        Parameters
        ----------
            image_l (np.array): Left grayscale image
            image_r (np.array): Right grayscale image
            matcher (str(): bm or sgbm): Stereo matcher
            NOTE: bm is faster than sgbm, but sgbm is more accurate

        Returns
        -------
            disparity_map (np.array): Disparity map [distance in pixels]
        """
        sad_window = 6 # Sum of absolute differences
        num_disparities = sad_window*16
        block_size = 11
        matcher_name = matcher
        number_of_image_channels = 1 # Grayscale -> 1, RGB -> 3

        # Compute disparity map
        if matcher_name == 'bm':
            matcher = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
        elif matcher_name == 'sgbm':
            matcher = cv2.StereoSGBM_create(minDisparity=0,
                                            numDisparities=num_disparities,
                                            blockSize=block_size,
                                            P1=8 * number_of_image_channels * sad_window ** 2,
                                            P2=32 * number_of_image_channels * sad_window ** 2,
                                            disp12MaxDiff=1,
                                            uniquenessRatio=10,
                                            speckleWindowSize=100,
                                            speckleRange=32,
                                            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

        disparity_map = matcher.compute(image_l, image_r).astype(np.float32) / 16.0

        return disparity_map

    def compute_depth_map(self, disparity_map, K_l, t_l, t_r):
        """
        Compute depth map of rectified camera.
        NOTE this is relative to the left camera as it is considered the world frame

        Parameters
        ----------
            disparity_map (np.array): Disparity map
            K_l (np.array): Left camera intrinsic matrix
            K_r (np.array): Right camera intrinsic matrix

        Returns
        -------
            depth_map (np.array): Depth map
        """

        # Compute baseline [meters]
        b = abs(t_l[0] - t_r[0])
        # Compute focal length [pixels]
        f = K_l[0,0]

        # NOTE Set the zero values and the -1 (No overlap between left and right camera image)
        # in disparity map to a small value to be able to divide with disparity. This will ensure that the
        # estimate depth of these points are very far away and thus can be ignored.
        disparity_map[disparity_map == 0.0] = 0.1
        disparity_map[disparity_map == -1.0] = 0.1

        # Calculate depth map
        depth_map = np.zeros(disparity_map.shape)
        depth_map = f * b / disparity_map

        return depth_map

    def stereo_to_depth(self, image_l, image_r, matcher):
        """
        Stereo to depth

        Parameters
        ----------
            image_l (np.array): Left image
            image_r (np.array): Right image
            matcher (str(): bm or sgbm): Stereo matcher
        Returns
        -------
            depth_map (np.array): Depth map
        """

        # Compute disparity map
        disp_map = self.compute_disparity_map(image_l, image_r, matcher)

        # Compute depth map
        depth_map = self.compute_depth_map(disp_map, self.K_l, self.t_l, self.t_r)

        return depth_map

    def ratio_test_filter(self, matches, ratio=0.45):
        """
        Filter matches based on distance ratio to closes features.
        If both matches are close then it is a bad feature and should be removed.
        NOTE Ratio test is used with a value of 0.45
        https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html

        Parameters
        ----------
            matches (list): List of matches -> List of 2 items list (KNN -> k = 2)
            ratio (float): Distance ratio

        Returns
        -------
            filtered_matches (list): List of goof filtered matches
        """
        # Initialize list for filtered matches
        filtered_matches = []

        # Filter matches based on distance
        for d1, d2 in matches:
            if d1.distance <= ratio * d2.distance:
                filtered_matches.append(d1)

        return filtered_matches

    def motion_estimation(self, matches_l, keypoints_l_prev, keypoints_l_curr, depth_map_prev, depth_threshold=1000):
        """
        Estimate motion from matched features

        Parameters
        ----------
            matches_l (list): List of matches between current and previous left image
            keypoints_l_prev (list): List of keypoints from previous (t-1) left image
            keypoints_l_curr (list): List of keypoints from current (t) left image
            depth_map_prev (np.array): Depth map from previous (t-1) left image
            depth_threshold (int): Depth threshold for motion estimation

        Returns
        -------
            motion (np.array): Motion vector
        """

        # Initialize rotation and translation vectors
        R_mat = np.eye(3)
        t_vec = np.zeros(3)

        # Get the 2D points from the matched features
        prev_image_points = np.float([keypoints_l_prev[match.queryIdx]].pt for match in matches_l)
        curr_image_points = np.float([keypoints_l_curr[match.trainIdx]].pt for match in matches_l)

        # Camera intrinsic parameters
        cx = self.K_l[0,2]
        cy = self.K_l[1,2]
        fx = self.K_l[0,0]
        fy = self.K_l[1,1]

        # Initialize points_object
        points_object = np.zeros((0, 3))
        # Initialize delete list for points that's depth is above threshold
        delete_list = []

        for i, (u, v) in enumerate(prev_image_points):
            # Get depth from depth map
            depth = depth_map_prev[int(v), int(u)]

            # Check if depth is above threshold
            if depth > depth_threshold:
                # Add index to delete list
                delete_list.append(i)
                continue
            else:
                # Compute 3D point
                x = ((u - cx)/fx) * depth
                y = ((v - cy)/fy) * depth
                z = depth
                points_object = np.vstack((points_object, np.array([x, y, z])))

        # Delete points that's depth is above threshold
        prev_image_points = np.delete(prev_image_points, delete_list, axis=0)
        curr_image_points = np.delete(curr_image_points, delete_list, axis=0)

        # Compute rotation and translation vectors from previous camera position to current camera position
        _, R_vec, t_vec, _ = cv2.solvePnPRansac(points_object, curr_image_points, self.K_l, None)

        # Convert rotation vector to rotation matrix
        R_mat, _ = cv2.Rodrigues(R_vec)

        return R_mat, t_vec





    # def triangulate_matched_features(self, matches, keypoints_l_prev, keypoints_l_curr):
    #     """
    #     Triangulate matched features

    #     Parameters
    #     ----------
    #         matches (list): List of matches
    #         keypoints_l_prev (list): List of keypoints from previous (t-1) left image
    #         keypoints_l_curr (list): List of keypoints from current (t) left image
    #     Returns
    #     -------
    #         point_3d (list): List of 3D points
    #     """
    #     # Initialize list for 3D points
    #     point_3d = []

    #     # Triangulate each matched feature
    #     for match in matches:
    #         # Get the keypoints for each match
    #         kp_l_prev = keypoints_l_prev[match.queryIdx].pt
    #         kp_l_curr = keypoints_l_curr[match.trainIdx].pt

    #         # Triangulate the 3D point
    #         point_4d = cv2.triangulatePoints(self.P_l, self.P_r, kp_l_prev, kp_r_prev) #3d prev point

    #         # Convert to homogeneous coordinates
    #         point_3d = point_4d[:3] / point_4d[3]

    #         # Add to list of 3D points
    #         point_3d.append(point_3d)

def main():
    """
    main function
    """
    SVO_dataset = VisualOdometry(dataset = "07")

    # Choose feature detector type
    detector = "orb"
    matcher = "bf"

    # Play images of the trip
    # vs.play_trip(SVO_dataset.image_l_list, SVO_dataset.image_r_list)

    """ Preform visual odometry on the dataset """

    # # Initialize the trajectory
    # estimated_traj = np.zeros((len(SVO_dataset.image_l_list), 3, 4))
    # T_current = np.eye(4) # Start at identity matrix
    # estimated_traj[0] = T_current[:3, :]

    # # Setup visual odometry images
    # image_l_curr = SVO_dataset.image_l_list[0]
    # image_r_curr = SVO_dataset.image_r_list[0]

    # """ TEST ONE RUN START """
    # # Previous image
    # image_l_prev = image_l_curr
    # image_r_prev = image_r_curr
    # # Current image
    # image_l_curr = SVO_dataset.image_l_list[1]
    # image_r_curr = SVO_dataset.image_r_list[1]

    # # Feature detection/extraction
    # keypoints_l_prev, descriptors_l_prev = SVO_dataset.feature_detection(detector, image_l_prev)
    # keypoints_r_prev, descriptors_r_prev = SVO_dataset.feature_detection(detector, image_r_prev)
    # keypoints_l_curr, descriptors_l_curr = SVO_dataset.feature_detection(detector, image_l_curr)
    # keypoints_r_curr, descriptors_r_curr = SVO_dataset.feature_detection(detector, image_r_curr)

    # # Feature matching
    # matches_l = SVO_dataset.feature_matching(matcher, detector, descriptors_l_prev, descriptors_l_curr)

    # NOTE this happens when stereo to depth is called
    # # Compute disparity map
    # disp_map = SVO_dataset.compute_disparity_map(image_l_prev, image_r_prev, 'sgbm')
    # # plt.figure(figsize=(11,7))
    # # plt.imshow(disp)
    # # plt.show()
    # # Compute depth map
    # depth_map = SVO_dataset.compute_depth_map(disp_map, SVO_dataset.K_l, SVO_dataset.t_l, SVO_dataset.t_r)
    # plt.figure(figsize=(11,7))
    # plt.imshow(depth_map)
    # plt.show()
    # # NOTE Plot depths as a histogram to see what depths range is and what can be filtered out
    # plt.hist(depth_map.flatten())
    # plt.show()

    # Stereo to depth
    # depth_map = SVO_dataset.stereo_to_depth(image_l_prev, image_r_prev, 'sgbm')
    # plt.figure(figsize=(11,7))
    # plt.imshow(depth_map)
    # plt.show()
    # NOTE Plot depths as a histogram to see what depths range is and what can be filtered out
    # plt.hist(depth_map.flatten())
    # plt.show()


    """ TEST ONE RUN END """

    xs = []
    ys = []
    zs = []

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=-20, azim=270)
    ax.plot(SVO_dataset.GT_poses[:, 0, 3], SVO_dataset.GT_poses[:, 1, 3], SVO_dataset.GT_poses[:, 2, 3], c = 'k') # Ground truth

    # Setup visual odometry images
    image_l_curr = SVO_dataset.image_l_list[0]
    image_r_curr = SVO_dataset.image_r_list[0]

    # Generate mask once
    depth_map = SVO_dataset.stereo_to_depth(image_l_curr, image_r_curr, 'sgbm')
    mask = SVO_dataset.create_mask(depth_map)

    # Loop through the images
    # for i in range(len(SVO_dataset.image_l_list) - 1):
    for i in range(2):
        # # Previous image
        image_l_prev = image_l_curr
        # image_r_prev = image_r_curr
        # Current image
        image_l_curr = SVO_dataset.image_l_list[i+1]
        image_r_curr = SVO_dataset.image_r_list[i+1]

        # Feature detection/extraction
        keypoints_l_prev, descriptors_l_prev = SVO_dataset.feature_detection(detector, image_l_prev, mask)
        keypoints_l_curr, descriptors_l_curr = SVO_dataset.feature_detection(detector, image_l_curr, mask)

        # Feature matching
        matches_l = SVO_dataset.feature_matching(matcher, detector, descriptors_l_prev, descriptors_l_curr)
        # print(f"Matches before filter: {len(matches_l)}")

        # Ratio test matches filtering
        matches_l = SVO_dataset.ratio_test_filter(matches_l)
        # print(f"Matches after filter: {len(matches_l)}")

        # Match visualize matches
        SVO_dataset.feature_match_visualize(keypoints_l_prev, keypoints_l_curr, image_l_prev, image_l_curr, matches_l)

        # Compute disparity map
        disp_map = SVO_dataset.compute_disparity_map(image_l_curr, image_r_curr, 'sgbm')
        # Make closer object light and further away object dark (Invert)
        disp_map /= disp_map.max()
        disp_map = 1 - disp_map # Invert colors
        disp_map = (disp_map*255).astype('uint8')
        disp_map = cv2.applyColorMap(disp_map, cv2.COLORMAP_RAINBOW) # Apply color

        # CURRENT POSITION OF CAR PLOTTED IN GREEN
        xs.append(SVO_dataset.GT_poses[i, 0, 3])
        ys.append(SVO_dataset.GT_poses[i, 1, 3])
        zs.append(SVO_dataset.GT_poses[i, 2, 3])

        # Plot moving path as car moves
        plt.plot(xs, ys, zs, c = 'chartreuse')
        plt.pause(0.000000000000000000000000001)
        cv2.imshow('camera', image_l_curr) # Play camera video
        cv2.imshow('disparity', disp_map)  # Play disparity video
        cv2.waitKey(1)

    plt.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
