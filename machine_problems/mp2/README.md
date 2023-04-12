# Morphological Operators
Morphological Operators were programmed to manipulate images by removing noise, enhancing the image shape and getting the boundary of a object
in the image.

### Five algorithms where programmed namely:
1) `Dilation:` 
    Dilation enhances the boundaries of objects and regions/gaps within objects. In this function, dilation is performed by adding the SE centered at
    each white pixel location in the input image. The result is a new binary image where white pixels have been added to the boundaries of the original
    objects, making them appear thicker or larger. 
    
2) `Erosion:` 
    Erosion reduces the size of an object by eroding away the boundary of the object. The function the image is first copied to a new image variable
    called "erosion". The function then loops through each pixel in the image and applies the SE to the pixel if the pixel is white. If the SE is not
    a subset of the object, i.e., if any pixel within the SE is black, the pixel in the erosion image corresponding to the current pixel is set to
    black. Otherwise, the pixel remains white.

3) `Opening:` 
    Opening first performs an erosion operation on the image using the provided structuring element. This operation removes small elements or noise in
    the image. Then, it performs a dilation operation on the eroded image using the same structuring element. This operation restores the shape of the
    remaining objects while smoothing their contours.
    
4) `Closing:` 
    Closing is used to remove small holes or gaps in a binary image. This is done by first dilating the image and then eroding it. The dilation step
    expands the white regions of the image, which fills in small gaps, while the erosion step shrinks the regions back down, preserving their original
    shape but closing the gaps. 
    
5) `Boundary:` 
    Boundary takes in an image and a structured element as arguments. It then creates a copy of the image and applies erosion using the provided
    structured element. Finally, it finds the boundary by subtracting the eroded image from the original image and returns the resulting boundary image.
    
### Results:
The Figure below displays the original images and all 4 Morphological function outputs on the original image. The image on the far right is the output
of the Boundary function that was applied on the Closing function output. Different SE where experimented with one is a 3x3 array and the other is a
Star structure with a 3x3 center array as in the class slides. Comparing Figure 1 Closing and Figure 2 Closing it is clear that the larger SE fills
the gaps in the original hand completely and thus a better SE than the 3x3 array. This also results in a better boundary. Other notes dilation and
closing generally fills in the gaps and makes the object in the images smoother where erosion and opening removes noise.

Figure 1: Morphological Operators with SE: 3x3 array
![3n3_Square_gun](https://user-images.githubusercontent.com/60977336/231596716-33d24673-573f-4ebb-8eac-0d07e1dbcaed.png)

Figure 2: Morphological Operators with SE: Star with a 3x3 center array
![Large_Star_gun](https://user-images.githubusercontent.com/60977336/231596748-c88e78ee-f60b-4fcc-9bcb-86083073f95c.png)

Figure 3: Morphological Operators with SE: 3x3 array
![3n3_Square_palm](https://user-images.githubusercontent.com/60977336/231596801-b76ad083-4cee-49f0-9e7e-9550b80c4900.png)

Figure 4: Morphological Operators with SE: Star with a 3x3 center array
![Large_Start_palm](https://user-images.githubusercontent.com/60977336/231596832-0d4d950a-f9d5-4fb2-88d1-2e152071b006.png)

