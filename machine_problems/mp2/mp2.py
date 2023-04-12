import cv2
import numpy as np

def main():

    # Load in images
    gun_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp2/test_images/gun.bmp', cv2.IMREAD_GRAYSCALE)
    palm_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp2/test_images/palm.bmp', cv2.IMREAD_GRAYSCALE)


    # Display images grid
    cv2.imshow('Test Image 1:', gun_img)
    cv2.imshow('Test Image 2:', palm_img)

    # Wait for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()