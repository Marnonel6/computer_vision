import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():

    # Load in images as BGR
    gun1_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp5/test_images/gun1.bmp', cv2.IMREAD_COLOR)
    joy1_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp5/test_images/joy1.bmp', cv2.IMREAD_COLOR)
    pointer1_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp5/test_images/pointer1.bmp', cv2.IMREAD_COLOR)
    lena_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp5/test_images/lena.bmp', cv2.IMREAD_COLOR)
    test1_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp5/test_images/test1.bmp', cv2.IMREAD_COLOR)
    # Put images next to each other
    test_images = cv2.hconcat([np.uint8(gun1_img), np.uint8(joy1_img), np.uint8(pointer1_img)])

    # Display images grid
    result_images = cv2.hconcat([np.uint8(gun1_img), np.uint8(joy1_img), np.uint8(pointer1_img)])
    # Final images
    final_images = cv2.vconcat([test_images, result_images])
    final_images2 = cv2.hconcat([np.uint8(lena_img), np.uint8(lena_img)])
    final_images3 = cv2.hconcat([np.uint8(test1_img), np.uint8(test1_img)])
    # Display
    cv2.imshow('Canny edge detection', final_images)
    cv2.imshow('Canny edge detection2', final_images2)
    cv2.imshow('Canny edge detection3', final_images3)
    # Wait for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
Detect human skin with a trained model

args:
    - human_skin_image_hsv: (cv2 - BGR image) BGR image of the desired picture used for detection
    - rgb_image: (cv2.imread(cv2.IMREAD_COLOR)) rgb image of the desired picture used for detection
    - trained_hist_model: (np.array()) Trained 2D histogram model in Blue and Green color space
return:
    - rgb_image: (cv2.imread(cv2.IMREAD_COLOR)) rgb image with only the skin color left
"""


if __name__ == '__main__':
    main()