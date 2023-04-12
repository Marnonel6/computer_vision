import cv2
import numpy as np

def main():

    # Load in images
    gun_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp2/test_images/gun.bmp', cv2.IMREAD_GRAYSCALE)
    palm_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp2/test_images/palm.bmp', cv2.IMREAD_GRAYSCALE)

    # 3x3 Square
    """
    1 1 1
    1 1 1
    1 1 1
    """
    # SE = [[-1,-1],[-1,0],[-1,1],
    #       [0 ,-1],[0 ,0],[0 ,1],
    #       [1 ,-1],[1 ,0],[1 ,1]]

    # Star
    """
    0 1 0
    1 1 1
    0 1 0
    """
    # SE = [[-1,0],[0,-1],[0,0],[0,1],[1,0]]

    # Star with 3x3 in middle
    """
    0 1 1 1 0
    1 1 1 1 1
    1 1 1 1 1
    1 1 1 1 1
    0 1 1 1 0
    """
    SE = [        [-2,-1],[-2,0],[-2,1],
          [-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2],
          [0 ,-2],[0 ,-1],[0 ,0],[0 ,1],[0 ,2],
          [1 ,-2],[1 ,-1],[1 ,0],[1 ,1],[1 ,2],
                  [2 ,-1],[2 ,0],[2 ,1]        ]

    # Star with 5x5 in middle
    """
    1 1 1 1 1
    1 1 1 1 1
    1 1 1 1 1
    1 1 1 1 1
    1 1 1 1 1
    """
    # SE = [[-2,-2],[-2,-1],[-2,0],[-2,1],[-2,2],
    #       [-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2],
    #       [0 ,-2],[0 ,-1],[0 ,0],[0 ,1],[0 ,2],
    #       [1 ,-2],[1 ,-1],[1 ,0],[1 ,1],[1 ,2],
    #       [2 ,-2],[2 ,-1],[2 ,0],[2 ,1],[2 ,2]]

    # Dilation
    gun_img_dilation = Dilation(gun_img, SE)
    palm_img_dilation = Dilation(palm_img, SE)

    # Erosion
    gun_img_erosion = Erosion(gun_img, SE)
    palm_img_erosion = Erosion(palm_img, SE)

    # Opening
    gun_img_opening = Opening(gun_img, SE)
    palm_img_opening = Opening(palm_img, SE)

    # Closing
    gun_img_closing = Closing(gun_img, SE)
    palm_img_closing = Closing(palm_img, SE)

    # Boundary
    gun_img_boundary = Boundary(gun_img_closing, SE)
    palm_img_boundary = Boundary(palm_img_closing, SE)

    # Display images grid
    gun = cv2.hconcat([np.uint8(gun_img), np.uint8(gun_img_dilation), np.uint8(gun_img_erosion), \
                       np.uint8(gun_img_opening), np.uint8(gun_img_closing), np.uint8(gun_img_boundary)])
    palm = cv2.hconcat([np.uint8(palm_img), np.uint8(palm_img_dilation), np.uint8(palm_img_erosion),\
                        np.uint8(palm_img_opening), np.uint8(palm_img_closing), np.uint8(palm_img_boundary)])

    # Display
    cv2.imshow(' 1: Normal                      2: Dilation                     3: Erosion\
                4: Opening                           5: Closing                 6: Boundary', gun)
    cv2.imshow('1: Normal                      2: Dilation                     3: Erosion\
                4: Opening                           5: Closing                 6: Boundary', palm)

    # Wait for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
Dilation of an image

args:
    - image: (cv2.IMREAD_GRAYSCALE) A bit map image
    - SE: (List) Structured element coordinates in a list
"""
def Dilation(image, SE):
    # Get image dimensions
    height, width = image.shape
    # Dilation image
    dilation = np.zeros((height, width))

    # Loop through image
    for u in range(height):
        for v in range(width):
            if image[u,v] == 255: # If pixel is white add the SE with it's centre at [u,v]
                for se in SE:
                    x = se[0] + u
                    y = se[1] + v
                    dilation[x,y] = 255

    return dilation

"""
Erosion of an image

args:
    - image: (cv2.IMREAD_GRAYSCALE) A bit map image
    - SE: (List) Structured element coordinates in a list
"""
def Erosion(image, SE):
    # Get image dimensions
    height, width = image.shape
    # Erosion images
    erosion = image.copy()

    # Loop through image
    for u in range(height):
        for v in range(width):
            # Subset flag to flag if SE is a subset of the object
            subset = True
            if image[u,v] == 255: # If pixel is white
                # Check if SE is a subset of the object if not then make [u,v] black
                for se in SE:
                    x = se[0] + u
                    y = se[1] + v
                    if height > x >= 0 and width > y >= 0: # Check if inside the images
                        if image[x,y] == 0: # Black pixel thus SE is not a subset of the object
                            subset = False
                # If SE is not a subset if the object then make the pixel black
                if subset == False:
                    erosion[u,v] = 0

    return erosion

"""
Opening of an image

args:
    - image: (cv2.IMREAD_GRAYSCALE) A bit map image
    - SE: (List) Structured element coordinates in a list
"""
def Opening(image, SE):

    # Opening image
    opening = image.copy()

    # Erosion then Dilation
    opening = Erosion(opening, SE)
    opening = Dilation(opening, SE)

    return opening

"""
Closing of an image

args:
    - image: (cv2.IMREAD_GRAYSCALE) A bit map image
    - SE: (List) Structured element coordinates in a list
"""
def Closing(image, SE):

    # Closing image
    closing = image.copy()

    # Dilation then Erosion
    closing = Dilation(closing, SE)
    closing = Erosion(closing, SE)

    return closing

"""
Get boundary of an object in an image

args:
    - image: (cv2.IMREAD_GRAYSCALE) A bit map image
    - SE: (List) Structured element coordinates in a list
"""
def Boundary(image, SE):

    # Boundary image
    erosion = image.copy()

    # Boundary = Image - Erosion(Image)
    erosion = Erosion(erosion, SE)
    boundary = image - erosion

    return boundary


if __name__ == '__main__':
    main()