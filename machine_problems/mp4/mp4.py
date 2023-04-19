import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():

    # Load in images
    gun1_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp4/test_images/gun1.bmp')
    joy1_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp4/test_images/joy1.bmp')
    pointer1_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp4/test_images/gun1.bmp')

    # Display images grid
    test_images = cv2.hconcat([np.uint8(gun1_img), np.uint8(joy1_img), np.uint8(pointer1_img)])
    # Display
    cv2.imshow('Test Images', test_images)
    # Wait for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()