from PIL import Image
import cv2
import matplotlib.pyplot as plt


def main():
    # face_image = Image.open('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp1')
    face_image = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp1/face.bmp')

    # plt.imshow(face_image)
    # plt.show()

    # Display image
    cv2.imshow('Image', face_image)

    # Key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# def CCL(image):

#     pass

if __name__ == '__main__':
    main()