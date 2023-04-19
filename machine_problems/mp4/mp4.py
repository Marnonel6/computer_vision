import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from PIL import Image
import os

def main():

    # Load in images as RGB
    gun1_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp4/test_images/gun1.bmp', cv2.IMREAD_COLOR)
    joy1_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp4/test_images/joy1.bmp', cv2.IMREAD_COLOR)
    pointer1_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp4/test_images/pointer1.bmp', cv2.IMREAD_COLOR)

    # 3D Hue and Saturation histogram
    # Open image using PIL
    img_pil = Image.open('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp4/train_images/hands/Hand_0000002.jpg')
    img_pil2 = Image.open('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp4/train_images/hands/Hand_0000092.jpg')

    # Train histogram model
    trained_hist_model = train_histogram()
    train_image_histogram(pointer1_img)

    # Convert RGB image to HSV color space
    gun1_hsv_img = cv2.cvtColor(gun1_img, cv2.COLOR_RGB2HSV)
    joy1_hsv_img = cv2.cvtColor(joy1_img, cv2.COLOR_RGB2HSV)
    pointer1_hsv_img = cv2.cvtColor(pointer1_img, cv2.COLOR_RGB2HSV)

    # Detect human skin color
    joy1_human_skin_img = detect_human_skin(joy1_hsv_img, joy1_img, trained_hist_model)

    # # Display images grid
    # test_images = cv2.hconcat([np.uint8(gun1_img), np.uint8(joy1_img), np.uint8(pointer1_img)])
    # Display
    # cv2.imshow('Test Images', test_images)
    cv2.imshow('Test Images', joy1_human_skin_img)
    # cv2.imshow('Train Images', train1_img)
    # Wait for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
Plot histogram of original image

args:
    - image: (cv2.IMREAD_GRAYSCALE) Gray scale image
"""
def train_image_histogram(image):
# def train_image_histogram(image1, image2):
    # # Histogram of pixel intensities
    # histogram, bins = np.histogram(image.ravel(), bins=256, range=[0, 256])

    # # Plot the histogram
    # plt.figure()
    # plt.plot(histogram, color='black')
    # # Set the axis labels and title
    # plt.xlabel('Pixel Intensity')
    # plt.ylabel('Frequency')
    # plt.title('Original Histogram of image intensities')
    # # Show plots
    # plt.show()



    # Set the range for Hue and Saturation
    h_range = [0, 180]
    s_range = [5, 256] # Set 5 as minimum to ignore the white background in Test images (White saturation = 0)

    # Set the number of bins for Hue and Saturation
    h_bins = 180
    s_bins = 256

    """ ONE IMAGE """
    # # Convert the jpg image to png format
    # png_image = image.convert('RGBA')

    # # Convert PIL image to NumPy array
    # img_np = np.array(png_image)

    # # Convert RGB to BGR (OpenCV uses BGR color format)
    # img_np = img_np[:, :, ::-1].copy()

    # # Convert BGR image to HSV
    # hsv_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)

    # cv2.imshow('Train Images', hsv_img)
    # # Wait for a key press to close the window
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Convert BGR image to HSV
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a 3D histogram (ONE IMAGE)
    hist, _, _ = np.histogram2d(hsv_img[:, :, 0].ravel(),
                                hsv_img[:, :, 1].ravel(),
                                bins=[h_bins, s_bins],
                                range=[h_range, s_range])

    # # Plot the 3D histogram
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(h_bins), np.arange(s_bins))
    ax.plot_surface(X, Y, hist.T, cmap='jet')
    ax.set_xlabel('Hue')
    ax.set_ylabel('Saturation')
    ax.set_zlabel('Count')
    plt.show()

    """ ONE IMAGE """

    """ TWO IMAGES """
    # # Convert the jpg image to png format
    # png_image1 = image1.convert('RGBA')

    # # Convert PIL image to NumPy array
    # img_np1 = np.array(png_image1)

    # # Convert RGB to BGR (OpenCV uses BGR color format)
    # img_np1 = img_np1[:, :, ::-1].copy()

    # # Convert the jpg image to png format
    # png_image2 = image2.convert('RGBA')

    # # Convert PIL image to NumPy array
    # img_np2 = np.array(png_image2)

    # # Convert RGB to BGR (OpenCV uses BGR color format)
    # img_np2 = img_np2[:, :, ::-1].copy()

    # # Convert BGR image to HSV
    # hsv_img1 = cv2.cvtColor(img_np1, cv2.COLOR_RGB2HSV)
    # hsv_img2 = cv2.cvtColor(img_np2, cv2.COLOR_RGB2HSV)

    # # Concatenate the Hue and Saturation channels of both images
    # hue_sat1 = np.concatenate((hsv_img1[..., 0], hsv_img1[..., 1]), axis=1)
    # hue_sat2 = np.concatenate((hsv_img2[..., 0], hsv_img2[..., 1]), axis=1)
    # hue_sat = np.concatenate((hue_sat1, hue_sat2), axis=0)

    # # Create a 3D histogram - (TWO IMAGES)
    # hist, _, _ = np.histogram2d(hue_sat[..., 0].ravel(),
    #                             hue_sat[..., 1].ravel(),
    #                             bins=[h_bins, s_bins],
    #                             range=[h_range, s_range])
    """ TWO IMAGES """

    """ TWO IMAGES """
    # # Convert BGR image to HSV
    # hsv_img1 = cv2.cvtColor(image1, cv2.COLOR_RGB2HSV)
    # hsv_img2 = cv2.cvtColor(image2, cv2.COLOR_RGB2HSV)

    # # Concatenate the Hue and Saturation channels of both images
    # hue_sat1 = np.concatenate((hsv_img1[..., 0], hsv_img1[..., 1]), axis=1)
    # hue_sat2 = np.concatenate((hsv_img2[..., 0], hsv_img2[..., 1]), axis=1)
    # hue_sat = np.concatenate((hue_sat1, hue_sat2), axis=0)

    # # Create a 3D histogram - (TWO IMAGES)
    # hist, _, _ = np.histogram2d(hue_sat[..., 0].ravel(),
    #                             hue_sat[..., 1].ravel(),
    #                             bins=[h_bins, s_bins],
    #                             range=[h_range, s_range])
    """ TWO IMAGES """

    # # Plot the 3D histogram
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # X, Y = np.meshgrid(np.arange(h_bins), np.arange(s_bins))
    # ax.plot_surface(X, Y, hist.T, cmap='jet')
    # ax.set_xlabel('Hue')
    # ax.set_ylabel('Saturation')
    # ax.set_zlabel('Count')
    # plt.show()



    """ ONE IMAGE """
    # # Convert the jpg image to png format
    # png_image = image1.convert('RGBA')

    # # Convert PIL image to NumPy array
    # img_np = np.array(png_image)

    # # Convert RGB to BGR (OpenCV uses BGR color format)
    # img_np = img_np[:, :, ::-1].copy()

    # # Convert BGR image to HSV
    # hsv_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)

    # # cv2.imshow('Train Images', hsv_img)
    # # # Wait for a key press to close the window
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()

    # # Create a 3D histogram (ONE IMAGE)
    # hist1, _, _ = np.histogram2d(hsv_img[:, :, 0].ravel(),
    #                             hsv_img[:, :, 1].ravel(),
    #                             bins=[h_bins, s_bins],
    #                             range=[h_range, s_range])
    """ ONE IMAGE """

    # # Plot the 3D histogram
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # X, Y = np.meshgrid(np.arange(h_bins), np.arange(s_bins))
    # ax.plot_surface(X, Y, hist1.T, cmap='jet')
    # ax.set_xlabel('Hue')
    # ax.set_ylabel('Saturation')
    # ax.set_zlabel('Count')
    # plt.show()

    # """ ONE IMAGE """
    # # Convert the jpg image to png format
    # png_image2 = image2.convert('RGBA')

    # # Convert PIL image to NumPy array
    # img_np2 = np.array(png_image2)

    # # Convert RGB to BGR (OpenCV uses BGR color format)
    # img_np2 = img_np2[:, :, ::-1].copy()

    # # Convert BGR image to HSV
    # hsv_img2 = cv2.cvtColor(img_np2, cv2.COLOR_BGR2HSV)

    # # cv2.imshow('Train Images', hsv_img)
    # # # Wait for a key press to close the window
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()

    # # Create a 3D histogram (ONE IMAGE)
    # hist2, _, _ = np.histogram2d(hsv_img2[:, :, 0].ravel(),
    #                             hsv_img2[:, :, 1].ravel(),
    #                             bins=[h_bins, s_bins],
    #                             range=[h_range, s_range])
    # """ ONE IMAGE """

    # # Plot the 3D histogram
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # X, Y = np.meshgrid(np.arange(h_bins), np.arange(s_bins))
    # ax.plot_surface(X, Y, hist2.T, cmap='jet')
    # ax.set_xlabel('Hue')
    # ax.set_ylabel('Saturation')
    # ax.set_zlabel('Count')
    # plt.show()

    # print(f"\n hist1 shape = {hist1.shape}")
    # print(f"\n hist2 shape = {hist2.shape}")

    # hist3 = hist1+hist2

    # print(f"\n hist3 shape = {hist3.shape}")

    # print(f"\n hist3 index [129][136] = {hist3[129][136]}")

    # # Plot the 3D histogram
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # X, Y = np.meshgrid(np.arange(h_bins), np.arange(s_bins))
    # ax.plot_surface(X, Y, hist3.T, cmap='jet')
    # ax.set_xlabel('Hue')
    # ax.set_ylabel('Saturation')
    # ax.set_zlabel('Count')
    # plt.show()


    # # Normalize the array
    # norm_arr = hist3 / np.max(hist3)

    # print(f"\n norm_arr index [129][136] = {norm_arr[129][136]}")

    # # Plot the 3D histogram
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # X, Y = np.meshgrid(np.arange(h_bins), np.arange(s_bins))
    # ax.plot_surface(X, Y, norm_arr.T, cmap='jet')
    # ax.set_xlabel('Hue')
    # ax.set_ylabel('Saturation')
    # ax.set_zlabel('Count')
    # plt.show()


"""
Plot histogram of original image

args:
    - image: (cv2.IMREAD_GRAYSCALE) Gray scale image
"""
# def train_image_histogram(image):
def train_histogram():

    # Set the range for Hue and Saturation
    h_range = [0, 180]
    s_range = [5, 256] # Set 5 as minimum to ignore the white background in Test images (White saturation = 0)

    # Set the number of bins for Hue and Saturation
    h_bins = 180
    s_bins = 256

    # First image flag
    Flag_first_img = True

    # Only load X amount of images
    training_size = 50
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
            image = Image.open(img_path)
            # Rest of the code to calculate histogram

            # image = Image.open(img_path)

            """ ONE IMAGE """
            # Convert the jpg image to png format
            png_image = image.convert('RGBA')

            # Convert PIL image to NumPy array
            img_np = np.array(png_image)

            # Convert RGB to BGR (OpenCV uses BGR color format)
            img_np = img_np[:, :, ::-1].copy()

            # Convert BGR image to HSV
            hsv_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)

            # cv2.imshow('Train Images', hsv_img)
            # # Wait for a key press to close the window
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # Create a 3D histogram (ONE IMAGE)
            hist, _, _ = np.histogram2d(hsv_img[:, :, 0].ravel(),
                                        hsv_img[:, :, 1].ravel(),
                                        bins=[h_bins, s_bins],
                                        range=[h_range, s_range])
            """ ONE IMAGE """

            # Plot the 3D histogram
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # X, Y = np.meshgrid(np.arange(h_bins), np.arange(s_bins))
            # ax.plot_surface(X, Y, hist.T, cmap='jet')
            # ax.set_xlabel('Hue')
            # ax.set_ylabel('Saturation')
            # ax.set_zlabel('Count')
            # plt.show()




            # print(f"\n hist shape = {hist.shape}")

            if Flag_first_img == True:
                # Total train data histogram
                train_hist = hist
                Flag_first_img = False
            else:
                train_hist += hist

            # print(f"\n train_hist shape = {train_hist.shape}")

            # print(f"\n hist3 index [129][136] = {hist3[129][136]}")

            # # Plot the 3D histogram
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # X, Y = np.meshgrid(np.arange(h_bins), np.arange(s_bins))
            # ax.plot_surface(X, Y, train_hist.T, cmap='jet')
            # ax.set_xlabel('Hue')
            # ax.set_ylabel('Saturation')
            # ax.set_zlabel('Count')
            # plt.show()

            # Only train on 'training_size' amount of images
            image_count += 1
            if image_count >= training_size:
                break

        else:
            print(f"File {img_path} does not exist. Skipping...")







    """ Loading with a for loop """
    # for i in range(2,1000):
    #     # Image does not exist flag
    #     Flag_no_image = False

    #     if i < 10:
    #         try:
    #             img_path = f"/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp4/train_images/hands/Hand_000000{i}.jpg"
    #         except FileNotFoundError:
    #             Flag_no_image = True
    #     elif i < 100:
    #         try:
    #             img_path = f"/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp4/train_images/hands/Hand_00000{i}.jpg"
    #         except FileNotFoundError:
    #             Flag_no_image = True
    #     elif i < 1000:
    #         try:
    #             img_path = f"/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp4/train_images/hands/Hand_0000{i}.jpg"
    #         except FileNotFoundError:
    #             Flag_no_image = True

    #     if Flag_no_image == False:
    #         image = Image.open(img_path)

    #         """ ONE IMAGE """
    #         # Convert the jpg image to png format
    #         png_image = image.convert('RGBA')

    #         # Convert PIL image to NumPy array
    #         img_np = np.array(png_image)

    #         # Convert RGB to BGR (OpenCV uses BGR color format)
    #         img_np = img_np[:, :, ::-1].copy()

    #         # Convert BGR image to HSV
    #         hsv_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)

    #         # cv2.imshow('Train Images', hsv_img)
    #         # # Wait for a key press to close the window
    #         # cv2.waitKey(0)
    #         # cv2.destroyAllWindows()

    #         # Create a 3D histogram (ONE IMAGE)
    #         hist, _, _ = np.histogram2d(hsv_img[:, :, 0].ravel(),
    #                                     hsv_img[:, :, 1].ravel(),
    #                                     bins=[h_bins, s_bins],
    #                                     range=[h_range, s_range])
    #         """ ONE IMAGE """

    #         # Plot the 3D histogram
    #         # fig = plt.figure()
    #         # ax = fig.add_subplot(111, projection='3d')
    #         # X, Y = np.meshgrid(np.arange(h_bins), np.arange(s_bins))
    #         # ax.plot_surface(X, Y, hist.T, cmap='jet')
    #         # ax.set_xlabel('Hue')
    #         # ax.set_ylabel('Saturation')
    #         # ax.set_zlabel('Count')
    #         # plt.show()




    #         # print(f"\n hist shape = {hist.shape}")

    #         if Flag_first_img == True:
    #             # Total train data histogram
    #             train_hist = hist
    #             Flag_first_img = False
    #         else:
    #             train_hist += hist

    #         # print(f"\n train_hist shape = {train_hist.shape}")

    #         # print(f"\n hist3 index [129][136] = {hist3[129][136]}")

    #         # # Plot the 3D histogram
    #         # fig = plt.figure()
    #         # ax = fig.add_subplot(111, projection='3d')
    #         # X, Y = np.meshgrid(np.arange(h_bins), np.arange(s_bins))
    #         # ax.plot_surface(X, Y, train_hist.T, cmap='jet')
    #         # ax.set_xlabel('Hue')
    #         # ax.set_ylabel('Saturation')
    #         # ax.set_zlabel('Count')
    #         # plt.show()


    # Normalize the array
    norm_arr = train_hist / np.max(train_hist)

    # print(f"\n norm_arr index [129][136] = {norm_arr[129][136]}")

    # Plot the 3D histogram
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(h_bins), np.arange(s_bins))
    ax.plot_surface(X, Y, norm_arr.T, cmap='jet')
    ax.set_xlabel('Hue')
    ax.set_ylabel('Saturation')
    ax.set_zlabel('Count')
    plt.show()

    return norm_arr

"""
Plot histogram of original image

args:
    - image: (cv2.IMREAD_GRAYSCALE) Gray scale image
"""
def detect_human_skin(human_skin_image_hsv, rgb_image, trained_hist_model):

    # Loop through image
    for i in range(human_skin_image_hsv.shape[0]):
        for j in range(human_skin_image_hsv.shape[1]):
            # Get pixel value at (i, j)
            saturation, hue = human_skin_image_hsv[i, j, 0], human_skin_image_hsv[i, j, 1]
            pixel_conf = trained_hist_model[saturation][hue]
            if pixel_conf > 0.01:
                continue
                # print(f"\n pixel SAT: {saturation}  HUE: {hue}")
                # print(f"\n pixel_conf: {pixel_conf}")
            else: # Make pixels black
                # print(f"\n rgb_image[i, j] before: {rgb_image[i, j]}")
                rgb_image[i, j] = [0, 0, 0]
                # print(f"\n rgb_image[i, j] after: {rgb_image[i, j]}")

    return rgb_image


if __name__ == '__main__':
    main()