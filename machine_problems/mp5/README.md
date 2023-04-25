# Canny Edge Detector from scratch
A Canny edge detector is a multi-stage algorithm that detects edges in images by applying a series of filters to reduce noise, calculate image gradients, suppress non-maximum edges, and using a high and low threshold with recursion to determine final edges.

Below is a description of the 6 functions used, in order, to implement canny edge detection:

1) `Gaussian smoothing:` This function preforms Gaussian smoothing on a gray scale image with a given kernel size and standard deviation. This reduces the noise in the image.

2) `Image gradient:` This function takes in the smoothed gray scale image and calculates the images gradient and magnitude.  Sobel operators are used to compute the horizontal and vertical edges of the image (Ix, Iy). The magnitude of the image gradient is calculated by taking the square root of the sum of the squares of the horizontal and vertical edges and the direction of the image gradient is calculated using the arctan2 of the vertical and horizontal edges.

3) `Compute thresholds:` This function computes high and low thresholds based on the magnitude image of the image gradient. First a histogram is created from the magnitude image and then the cumulative histogram is generated and normalized. The first bin of the cumulative distribution that is greater than or equal to the percentage of non-edge pixels is determined to be the high threshold. The low threshold is calculated by multiplying the high threshold with the given ratio.

4) `Non-maximum suppression:` This is used to thin out the edges obtained from applying gradient-based edge detection. It works by iterating through each pixel in the magnitude image and comparing the gradient direction of the pixel to its neighboring pixels. The output is obtained by only keeping pixels whose magnitude is greater than or equal to its two neighboring pixels in the direction of the gradient.

5) `Find edges:` The algorithm loops through the suppressed image and assigns pixel values to each image based on the high and low threshold values. Pixels with values above the high threshold are considered strong edges, those between the high and low thresholds are considered weak edges, and those below the low threshold are not considered edges. The output images have different pixel values to differentiate between strong and weak edges.

6) `Edge linking:` The input to the function is an image containing both strong (255) and weak (125) edges. The function loops through each pixel in the image and checks if the pixel is a weak edge. If it is, it checks if any of its eight neighboring pixels has a strong edge. If it does, the weak edge is made strong by setting its value to 255. If not, the weak edge is deleted by setting its value to 0. The function returns an image containing only strong edges.

### Results:
Figure 1 displays the output image of each function described above. (Figure 1: Top, Middle left) The Gaussian smoothing makes the image less sharp and thus reduces the edge noise. (Figure 1: Top, Middle right) The edges is then found by calculating the gradient of the image. The image magnitude displays the approximate edge locations. (Figure 1: Top, Right)  Next the Non-Maxima suppression makes the edges in the magnitude tinner and thus the edges more clear.  (Figure 1: Bottom left 3 images) The high and low threshold is then applied to the image and the edges are categorized in either strong or weak edges. (Figure 1: Bottom right) Lastly the weak edges are used to connect and extend the strong edges. This is the final result from the Canny edge detection algorithm.

`Figure 1: Image at each step in the canny edge detection algorithm.`

![Screenshot from 2023-04-23 19-15-03](https://user-images.githubusercontent.com/60977336/234143612-e0e827c4-84a0-44df-9021-876925f9259e.png)

Different high and low thresholds where experimented with. Decreasing the high threshold increases the amount of edges that are considered strong edges and thus drastically increases the edges and noise in the image. Decreasing the low threshold increases the amount of weak edges that can connect strong edges to generate clean lines. Thus a good choice is a high threshold that does not capture noise, but still captures parts of the main features of an image and then a low threshold that captures all the desired features in an image with some noise.

Figure 2 displays edge detectors that are in the OpenCV package. It is evident that Canny edge detection outperforms the other detection methods in accuracy and noise reduction. The canny edge detection that was implemented from scratch displayed in Figure 1 is comparable in accuracy to the builtin Canny edge detection algorithm in OpenCV.

`Figure 2: OpenCV's built in edge detection algorithms

![Screenshot from 2023-04-23 19-16-38](https://user-images.githubusercontent.com/60977336/234143630-d1c4b523-40ec-4c6c-b7eb-d3a762dfb2e7.png)

### Other test images:
![Screenshot from 2023-04-23 19-15-38](https://user-images.githubusercontent.com/60977336/234143827-edf7d54c-0f19-4226-893f-064c75edc140.png)

![Screenshot from 2023-04-23 19-12-50](https://user-images.githubusercontent.com/60977336/234143804-e463c748-a540-47b1-91e8-241d1d925c23.png)

![Screenshot from 2023-04-23 19-13-27](https://user-images.githubusercontent.com/60977336/234143808-93431306-cb5b-43e8-b244-23d04c91c9bb.png)

![Screenshot from 2023-04-23 19-14-11](https://user-images.githubusercontent.com/60977336/234143823-fa587e30-1a42-4f3e-8f82-1f6c2c016f45.png)












