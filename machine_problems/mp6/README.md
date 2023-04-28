# Hough Transform algorithm for line ditection
The Hough transform is a computer vision technique used to detect straight lines in an image. It works by converting the image space to a parameter space, where each point represents a line in the original image. The transform then accumulates points in the parameter space for each edge pixel in the image. The resulting accumulation will have peaks that correspond to the parameters of the straight lines in the image. The Hough transform can handle noise, gaps, and partial occlusion in the image, making it a popular technique for line detection in computer vision.

Below is a description of the 4 main steps used to implement a Hough Transform line detection algorithm:

1) `Sobel edge detection:` Sobel operators are used to compute the horizontal and vertical edges of the image (Ix, Iy). The magnitude of the image gradient is calculated by taking the square root of the sum of the squares of the horizontal and vertical edges and the direction of the image gradient is calculated using the arctan2 of the vertical and horizontal edges. The magnitude is also suppressed by filtering out low magnitude values according to a input threshold

2) `Hough transform:` The Hough Transform uses an edge-detected image (gradient magnitude) to vote for lines in polar space (rho, theta), where rho represents the distance from the origin to the line and theta represents the angle between the x-axis and the normal to the line. The function takes an input image and a threshold value to filter out edges with low gradient magnitude. It then initializes the maximum values of rho and theta based on the dimensions of the image and scales theta to match the range of rho. The function then loops through the image and for each edge pixel, it votes for lines in the polar space by incrementing the values of the corresponding (rho, theta) coordinates. This formula `rho = x*cos(theta) + y*sin(theta)` is used to convert to polar coordinates. The result is a polar space with votes for lines, which can be further processed to extract the lines in the original image. Histogram equalization is preformed on the resulting polar space to make it more visible when displayed. The function returns the polar space, scaling factor, and maximum values of rho and theta for further analysis.

3) `K-means clustering:` The Hough Transform output/polar space images are filtered to keep only the pixels with higher votes by using threshold values. This reveals the dominant clusters. The data is then manipulated to work with the Kmeans algorithm from the sklearn library. The Kmeans algorithm is then applied to the data with a specified number of clusters. Finally, the cluster centroids for each image are obtained from the Kmeans algorithm. These centroids [rho, theta] represent a line in the origin image x,y space.

4) `Draw the predicted lines:` Using the cluster centroids obtained from K-means clustering the predicted lines is then drawn onto the original image. The number of lines drawn correspond to the number of clusters specified to the Kmeans algorithm.

### Results:

Figure 1 displays the output of a image at each step described above. The figure at the top right of Figure 1 displays the votes cased by the Sobel magnitude edges. Five clusters can be seen in the image with one cluster at the bottom displaying a wide spread. This is the hand line cluster. The filtered magnitude is displayed at the bottom left of the image. Here the clusters are more clear and the Kmeans algorithm can easily find the centroids. Finally, at the bottom right of Figure 1 the predicted lines are overlaid on top of the original image.

`Figure 1: Image at each step in the implementation of a Hough transform algorithm`

![input_result](https://user-images.githubusercontent.com/60977336/235040508-14c2cb18-7aa1-44f2-9b93-0f1798978d71.png)

![test2_result](https://user-images.githubusercontent.com/60977336/235040534-8825c7cc-7f75-4537-9b17-9d9e4833eb3b.png)

![test_result](https://user-images.githubusercontent.com/60977336/235040541-7e837790-73c2-4d20-851b-9f8db54be223.png)

