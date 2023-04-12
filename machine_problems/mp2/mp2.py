import cv2
import numpy as np

def main():

    # Load in images
    gun_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp2/test_images/gun.bmp', cv2.IMREAD_GRAYSCALE)
    palm_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp2/test_images/palm.bmp', cv2.IMREAD_GRAYSCALE)

    SE = [[-1,-1],[-1,0],[-1,-1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]

    # for se in SE:
    #     x = se[0] + 1
    #     y = se[1] + 1
    #     print(f"\n se: {x,y}")

    gun_img_dilation = Dilation(gun_img, SE)

    # Display images grid
    cv2.imshow('gun_img:', gun_img)
    cv2.imshow('gun_img_dilation:', gun_img_dilation)

    # Wait for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()



'''  '''
def Dilation(image, SE): # SE -> Structured Element
    # Get image dimensions
    height, width = image.shape
    # Which pixels to change: 255 - White, 0 - Black
    dilation = np.zeros((height, width))

    # Loop through image
    for u in range(height):
        for v in range(width):
            if image[u,v] == 255: # If pixel is white add the SE with it's centre at [u,v]
                for se in SE:
                    # print(f"\n se: {se}")
                    x = se[0] + u
                    y = se[1] + v
                    dilation[x,y] = 255

    return dilation




if __name__ == '__main__':
    main()