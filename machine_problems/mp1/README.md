# Image segmentation from scratch
A `sequential connected component labeling` algorithm is used for image segmentation to identify and assign a unique label to each connected 
components or objects in the image. The `size filter` is then applied to the labeled image to filter out components based on their size (number of pixels).
The gray scale images are loaded into Python with OpenCV as numpy arrays. 

### Two algorithms where programmed namely:
1) `Sequential Connected Component Labeling:` 
    For this algorithm the image is scanned from left to right and top to bottom to find and label objects. If a pixel is white (255) the algorithm 
    tries to label it based on its neighboring pixels with the following rules:
    - If the upper and left neighboring pixels both are unlabeled (0), thus not part of an object (0, Black) then the white pixel is considered the
      start of a new object and the current label counter is increase and the pixel is then labeled. The label counter starts at 0, thus the first
      white pixel will be assigned a label of 1.
    - If the upper and left neighboring pixels both have the same non-zero label then we assign the upper label to the current white pixel as they
      are connected.
    - If the upper neighboring pixel has a non-zero label and the left neighboring pixel has a zero label then the maximum between the two, which
      is the upper neighboring pixel label is assigned to the current pixel as it is connected.
    - If the left neighboring pixel has a non-zero label and the upper neighboring pixel has a zero label then the maximum between the two, which
      is the left neighboring pixel label is assigned to the current pixel as it is connected.
    - Lastly, if both the upper and left neighboring pixels are non-zero then the current pixel is assigned the upper pixels label and an equivalence
      table is created with the upper and left neighboring pixels labels to indicate that they are part of the same object.
	
      After all the pixels have been labeled, the algorithm iterates over the equivalence tables to 	combine overlapping equivalence tables into one table per object. Then the algorithm assigns 	different intensities to each object based on the number of equivalence tables. Finally, the 	labeled image is returned.
    
2) `Size Filter:` 
    The algorithm is a simple filter that filters objects in an image based on their size. The algorithm works by iterating over each pixel in the
    input image and checking whether the pixel belongs to an object that is smaller than the specified size. If the object is smaller, the pixel is
    set to black (0), effectively removing the object from the image. The filtered image is returned.
    
### Results:
Figure below displays the gray scale input images. From left to right a persons face is seen, then a human hand in the form of a gun with noise and
lastly an abstract test shape.

![image](https://user-images.githubusercontent.com/60977336/230953242-1d0779e3-41b2-4d19-8ce6-7159b5ee57f3.png)

Figure below displays the results after the sequential connected component labeling was performed. The figure shows that in each image the individual
objects was discovered and labeled with a unique pixel intensity. This confirms that the algorithm correctly identified the individual objects.

![image](https://user-images.githubusercontent.com/60977336/230953288-58c6df19-83b9-435a-838d-645bbcd3a67d.png)

The human hand in the form of a gun had unwanted noise around it which can be seen in the Figure above. This was filtered out with the size filter
algorithm. The Figure below displays the filtered out hand.

![image](https://user-images.githubusercontent.com/60977336/230953307-2d7445cf-ced1-4e55-b183-92eba196251d.png)
