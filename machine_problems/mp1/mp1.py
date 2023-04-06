from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Load in images
    face_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp1/face.bmp', cv2.IMREAD_GRAYSCALE)
    gun_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp1/gun.bmp', cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp1/test.bmp', cv2.IMREAD_GRAYSCALE)
    print(test_img)

    labels = seq_CCL(face_img)

    # Resize the image to 500x500 pixels
    labels = cv2.resize(labels, (500, 500))

    cv2.imshow('labels', np.uint8(labels))

    # labels1 = np.zeros((5, 5)) # Lables for each pixel
    # labels1[1,1] = 1
    # labels1[1,2] = 100
    # labels1[1,3] = 150
    # labels1[1,3] = 200
    # labels1[4,4] = 255
    # cv2.imshow('labels1', np.uint8(labels1))


    # print(face_img)

    # # Display image
    # cv2.imshow('face_img', face_img)
    # cv2.imshow('gun_img', gun_img)
    # cv2.imshow('test_img', test_img)

    # Key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def seq_CCL(image):
    # Get image dimensions
    height, width = image.shape
    labels = np.zeros((height, width)) # Lables for each pixel
    L_num = 1 # Label/Clustem number/identifier

    for u in range(height):
        for v in range(width):
            if image[u,v] == 255: # If pixel is white
                Lu = labels[u-1,v] # Upper label
                Ll = labels[u,v-1] # Left label
                if Lu == Ll and Lu != 0 and Ll != 0: # Thus Upper and left is the same label
                    labels[u,v] = Lu
                    # print("\n Same label")
                elif Lu != Ll and not (Lu and Ll):   # Either is 0
                    labels[u,v] = max(Lu, Ll)
                    # print("\n One Zero")
                elif Lu != Ll and Lu > 0 and Ll > 0: # Both has labels and an item should be added to equal table
                    labels[u,v] = min(Lu, Ll)
                    E_table(Lu, Ll)                  # Set these equal to each other in the equal table
                    # print("\n Both Not Zero")
                else:
                    L_num += 25
                    labels[u,v] = L_num
                    print(L_num)
                    # print("\n New Cluster")

    return labels


def E_table(Lu, Ll):
    pass



if __name__ == '__main__':
    main()