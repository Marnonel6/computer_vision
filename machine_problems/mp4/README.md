# Histogram-based Skin Color Detection
A `Histogram-based skin color detection algorithm` was programmed to extract human skin color from an image.

A 2D histogram was created by converting the BGR image to HSV. The 2D histogram is created by counting the frequency of hue and saturation pairs in
a dataset with training images. The histogram was then normalized. When looking at the trained histogram model it was evident that white (Saturation = 0)
was in the vast majority as all other hue and saturation pairs was suppressed after normalizing. This made sense as the training images where all of
hands on a white background. A filter was then added when training the histogram to not include saturation value under 25 as to filter out white.
The new trained 2D histogram had a cluster between Hue[0,25] and Saturation[40,140]. This was the skin color cluster.

When testing the trained histogram model on test images the images was converted to HSV and then each pixel was iterated through. The Hue and Saturation
values where used to index the 2D histogram model and when the normalized value at the index was greater than 0.1 the pixel was classified to be skin
color otherwise the pixel was set to black. This ensured that the output image will only contain pixels that is associated with skin color. 

The 2D histogram model was trained on 10, 250 and 1000 images with no significant accuracy increase after 250 images. The model was saved and can now
be loaded and used.

The BGR color space was also used for training a 2D histogram skin color detection model, but the accuracy was low. The HSV color space was chosen as
it had the highest accuracy.

### Results:
Figure 1 displays the trained 2D histogram trained with a 1000 images in the HSV color space. The skin color cluster is evident between Hue[0,25] and
Saturation[40,140].

`Figure 1: HSV color space 2D histogram trained model on a 1000 images`

![10_train_images_Skin_trained_model](https://user-images.githubusercontent.com/60977336/233508980-32e78337-c68d-4520-8941-9296bf00a3dd.png)

Figure 2 displays the test images and the results of the skin color detection using the HSV space with a trained 2D histogram model. The skin color
is identified with good accuracy. [Closing](https://github.com/Marnonel6/computer_vision/tree/main/machine_problems/mp2) can be preformed to fill up
the holes. Also [size/noise filtering](https://github.com/Marnonel6/computer_vision/tree/main/machine_problems/mp1) can be used to remove the small
noise in the output image.

`Figure 2: (Top) - Input test images. (Bottom) - Results with HSV trained 2D histogram model`

![Human skin color detection 1000 Train images](https://user-images.githubusercontent.com/60977336/233509075-19a800e7-f84b-4124-ac51-45e06a058327.png)

Figure 3 displays the trained 2D histogram trained with a 100 images in the BGR color space. Possible black and white pixels where filtered out by
adding a only considering the range of [10-250]. The skin color cluster is vague in the histogram and thus the model accuracy is low.

`Figure 3: BGR color space 2D histogram trained model on a 100 images`

![BGR_10_train_images_Skin_trained_model](https://user-images.githubusercontent.com/60977336/233509149-ac97111f-f3f0-4f94-b316-cc946dd519d3.png)

Figure 4 displays the test images and the results of the skin color detection using the BGR space with a trained 2D histogram model. The skin
color is identified with a low accuracy. BGR is thus not a good color space for skin color detection and thus HSV is chosen for the final model.

`Figure 4: (Top) - Input test images. (Bottom) - Results with BGR trained 2D histogram model`

![RGB_10_train_images_Skin_trained_model](https://user-images.githubusercontent.com/60977336/233509225-a1d27d77-00c3-47bd-98db-7bff16eba28b.png)


Figure 5 displays a sample form the training dataset that was used.

`Figure 5: Training dataset sample`

![image](https://user-images.githubusercontent.com/60977336/233509258-82c02163-d734-4a7e-b0ed-deb9a05714bc.png)







