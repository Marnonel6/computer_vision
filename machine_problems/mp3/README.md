# Histogram Equalization
A Histogram equalization algorithm was programmed to improve the contrast and brightness of an image. It works by redistributing the pixel values of an
input image over the entire range of available intensities. This improves the contrast of the image and makes image appear more vivid by highlighting
details in the image.

The main function programmed is `histogram_equalization()`. This function first calculates the histogram of the input image using the `np.histogram()`
function and then calculates the cumulative distribution of the histogram using the function `np.cumsum()`. The cumulative histogram is then normalized
by dividing it by the maximum value of the cumulative histogram.

Next, the function makes a copy of the input image and applies histogram equalization on each pixel. This is done by using the original image to index
the normalized cumulative distribution values and then multiplying by the output gray level, L2/255, and then saving it to the copy of the input image.
This is also know as the Transfer function. This redistributes the pixel values as described in the first paragraph.

`Left - Input image of the moon. Right - Histogram equalization output image.`

![ 1: Normal                                                2: Improved_screenshot_16 04 2023](https://user-images.githubusercontent.com/60977336/232349791-8f77a911-d066-4eb1-ae26-9bf1cc66b91e.png)
   
### Results:
Figure 1 displays the input image histogram. From the image it is evident that the majority of this image’s pixel intensities is concentrated between
110-130. This make the image look faded and hard to distinguish features in the image.

`Figure 1: Input image histogram`

![Original_Image_Histogram](https://user-images.githubusercontent.com/60977336/232349646-b525c99c-2da6-45d3-9e86-93965e7bad8a.png)

Figure 2 displays the normalized cumulative distribution of the input image’s histogram. This is also a  visual way of displaying the transfer function
that is used for histogram equalization.

`Figure 2: Normalized cumulative distribution`

![Cumulative_histogram_distribution_TransferFunction](https://user-images.githubusercontent.com/60977336/232349658-accbdf45-4a4c-44ec-a416-8a74b1f01407.png)

Figure 3 displays the histogram of the input image (Histogram, Figure 1) after the histogram equalization is preformed. It is evident that the pixel
intensities is spread over the entire available intensity range and thus the image is more vivid.

`Figure 3: Histogram equalization image histogram`

![Image_Histogram_after_histogram_equalization](https://user-images.githubusercontent.com/60977336/232349667-825e36be-ab4b-4f19-b0ec-ae59414d08f8.png)

Figure 4 displays a image of the moon before and after histogram equalization. It is clear that after histogram equalization the image has more contrast
and thus the features in the image is more clear. Next lighting correction can be performed to improve the image with linear or quadratic plane fitting.

`Figure 4: (Left) Input image. (Right) Histogram equalization output image`

![Screenshot from 2023-04-16 17-15-58](https://user-images.githubusercontent.com/60977336/232349768-428debf9-525f-414b-b4f4-6e27f04a631e.png)

