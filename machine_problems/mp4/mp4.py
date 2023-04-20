import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from PIL import Image
import os

def main():

    # Load in images as BGR
    gun1_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp4/test_images/gun1.bmp', cv2.IMREAD_COLOR)
    joy1_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp4/test_images/joy1.bmp', cv2.IMREAD_COLOR)
    pointer1_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp4/test_images/pointer1.bmp', cv2.IMREAD_COLOR)
    # Put images next to each other
    test_images = cv2.hconcat([np.uint8(gun1_img), np.uint8(joy1_img), np.uint8(pointer1_img)])

    # Convert BGR image to HSV color space
    gun1_hsv_img = cv2.cvtColor(gun1_img, cv2.COLOR_BGR2HSV)
    joy1_hsv_img = cv2.cvtColor(joy1_img, cv2.COLOR_BGR2HSV)
    pointer1_hsv_img = cv2.cvtColor(pointer1_img, cv2.COLOR_BGR2HSV)

    """ 1) Train histogram model (Train this and then comment this line out and just load trained model) """
    trained_hist_model = train_histogram()

    """ 2) Load trained model """
    trained_hist_model = np.load("trained_hist_model.npy")

    # Detect human skin color
    gun1_human_skin_img = detect_human_skin(gun1_hsv_img, gun1_img, trained_hist_model)
    joy1_human_skin_img = detect_human_skin(joy1_hsv_img, joy1_img, trained_hist_model)
    pointer1_human_skin_img = detect_human_skin(pointer1_hsv_img, pointer1_img, trained_hist_model)

    # Display images grid
    result_images = cv2.hconcat([np.uint8(gun1_human_skin_img), np.uint8(joy1_human_skin_img), np.uint8(pointer1_human_skin_img)])
    final_images = cv2.vconcat([test_images, result_images])
    # Display
    cv2.imshow('Human skin color detection', final_images)
    # Wait for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
Get a 2D histogram on a training dataset of images with hue and saturation on the two axis.
"""
def train_histogram():

    # First image flag
    Flag_first_img = True

    # Only load X amount of images
    training_size = 100
    image_count = 0

    """ Using directory """
    # Define the directory containing the training images
    img_dir = '/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp4/train_images/hands/'
    # Define the list of images to use for training
    img_list = []

    # Loop over all files in the directory and append to the list if it is an image file
    for filename in os.listdir(img_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_list.append(os.path.join(img_dir, filename))

    # Loop over all images and calculate the histogram
    for img_path in img_list:
        if os.path.exists(img_path):

            image = cv2.imread(img_path)
            # Convert BGR image to HSV
            hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            """ 1.1 START - Very fast but complicated, because of range values change """
            # Set the range for Hue and Saturation
            s_range = [25, 256] # Set 25 as minimum to ignore the white/shadow gray background in Test images (White saturation = 0)
            # NB now saturation value of 25 corresponds to 0 index in histogram
            h_range = [0, 256]
            h_bins = 257
            s_bins = 257

            # Create a 2D histogram
            hist, _, _ = np.histogram2d(hsv_img[:, :, 0].ravel(),
                                        hsv_img[:, :, 1].ravel(),
                                        bins=[np.arange(0, h_bins), np.arange(25, s_bins)],
                                        range=[h_range, s_range])
            """ 1.1 END """

            """ 1.2 START - Very slow use only 10 training images max - Shantao"""
            # Hue = []
            # Saturation = []
            # img_array = np.array(hsv_img)
            # x = np.shape(img_array)[0]-1
            # y = np.shape(img_array)[1]-1
            # for i in range(x+1):
            #     for j in range(y+1):
            #         if img_array[i][j][1]>25:
            #             Hue.append(img_array[i][j][0])
            #             Saturation.append(img_array[i][j][1])

            # # Create a 2D histogram (ONE IMAGE)
            # hist, _, _ = np.histogram2d(H, S, bins=[np.arange(0, 257), np.arange(0, 257)])
            """ 1.2 END """

            if Flag_first_img == True:
                # Total train data histogram
                train_hist = hist
                Flag_first_img = False
            else:
                train_hist += hist

            # Only train on 'training_size' amount of images
            image_count += 1
            if image_count >= training_size:
                break


    # Normalize trained dataset model
    norm_trained_model = train_hist / np.max(train_hist)

    # Save model
    np.save("trained_hist_model.npy", norm_trained_model)

    # # Plot the 3D histogram
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # X, Y = np.meshgrid(np.arange(h_bins), np.arange(s_bins))
    # ax.plot_surface(X, Y, norm_trained_model.T, cmap='jet')
    # ax.set_xlabel('Hue')
    # ax.set_ylabel('Saturation')
    # ax.set_zlabel('Count')
    # plt.show()

    # Plot the 2D histogram
    plt.imshow(norm_trained_model)
    plt.colorbar()
    plt.show()

    return norm_trained_model

"""
Detect human skin with a trained model

args:
    - human_skin_image_hsv: (cv2 - HSV image) HSV image of the desired picture used for detection
    - rgb_image: (cv2.imread(cv2.IMREAD_COLOR)) RGB image of the desired picture used for detection
    - trained_hist_model: (np.array()) Trained 2D histogram model
return:
    - rgb_image: (cv2.imread(cv2.IMREAD_COLOR)) RGB image with only the skin color left
"""
def detect_human_skin(human_skin_image_hsv, rgb_image, trained_hist_model):

    # Loop through image
    for i in range(human_skin_image_hsv.shape[0]):
        for j in range(human_skin_image_hsv.shape[1]):
            # Get pixel value at (i, j)
            hue, saturation = human_skin_image_hsv[i, j, 0], human_skin_image_hsv[i, j, 1]
            saturation -= 25
            if saturation > 0:
                pixel_conf = trained_hist_model[hue][saturation]
                if pixel_conf > 0.1:
                    continue
                else: # Make pixels black
                    rgb_image[i, j] = [0, 0, 0]
            else:
                rgb_image[i, j] = [0, 0, 0]

    return rgb_image


if __name__ == '__main__':
    main()