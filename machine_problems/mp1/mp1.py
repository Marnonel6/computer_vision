import cv2
import numpy as np

def main():
    # Load in images
    face_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp1/test_images/face.bmp', cv2.IMREAD_GRAYSCALE)
    gun_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp1/test_images/gun.bmp', cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread('/home/marno/Classes/Spring23/CV/computer_vision/machine_problems/mp1/test_images/test.bmp', cv2.IMREAD_GRAYSCALE)

    # Preform Sequential connected component labeling
    labels_face = seq_CCL(face_img)
    labels_gun = seq_CCL(gun_img)
    labels_test = seq_CCL(test_img)

    # Resize images to the same size
    # Original images
    orginal_face_img = cv2.resize(face_img, (250, 250))
    orginal_gun_img = cv2.resize(gun_img, (250, 250))
    orginal_test_img = cv2.resize(test_img, (250, 250))
    # Images after CCL
    labels_face = cv2.resize(labels_face, (250, 250))
    labels_gun = cv2.resize(labels_gun, (250, 250))
    labels_test = cv2.resize(labels_test, (250, 250))
    # Filted images
    # Size filter on pixel amount
    labels_gun_filter = size_filter(labels_gun, 224)
    labels_gun_filter = cv2.resize(labels_gun_filter, (250, 250))

    # Combine the images into three rows
    hconcat1 = cv2.hconcat([np.uint8(orginal_face_img), np.uint8(orginal_gun_img), np.uint8(orginal_test_img)])
    hconcat2 = cv2.hconcat([np.uint8(labels_face), np.uint8(labels_gun), np.uint8(labels_test)])
    hconcat3 = cv2.hconcat([np.uint8(labels_face), np.uint8(labels_gun_filter), np.uint8(labels_test)])
    # Stack the three rows in a grid
    final_image = cv2.vconcat([hconcat1,hconcat2,hconcat3])
    # Display images grid
    cv2.imshow('[Row1] - Original, [Row2] - Sequential connected component labeling, [Row3] - Size filter', final_image)

    # Wait for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

''' Sequential connected component labeling '''
def seq_CCL(image):
    # Get image dimensions
    height, width = image.shape
    labels = np.zeros((height, width)) # Lables for each pixel
    L_num = 0
    # Empty dictionary to hold all the sets/equivalence tables
    number_sets = {}

    for u in range(height):
        for v in range(width):
            if image[u,v] == 255: # If pixel is white
                Lu = labels[u-1,v] # Upper label
                Ll = labels[u,v-1] # Left label
                if Lu == Ll and Lu != 0 and Ll != 0: # Thus Upper and left is the same label
                    labels[u,v] = Lu

                elif Lu != Ll and not (Lu and Ll): # Either is 0
                    labels[u,v] = max(Lu, Ll)

                elif Lu != Ll and Lu > 0 and Ll > 0: # Both has labels and an item should be added to equivalence tables
                    labels[u,v] = min(Lu, Ll)
                    # Set these equal to each other in the equivalence tables
                    number_sets = E_table(Lu, Ll, number_sets)

                    # This is an easy way instead of using sets
                    # if Lu == labels[u,v]:
                    #     Ll = Lu
                    # else:
                    #     Lu = Ll

                else:  # Increase label if new object is discovered
                    L_num += 1
                    labels[u,v] = L_num

    # Overlapping keys to remove
    to_remove_set = set()
    for key1, set1 in number_sets.items():
        for key2, set2 in number_sets.items():
            if key1 != key2 and len(set1.intersection(set2)) > 0:
                # Combine the two sets and add the result to set1
                set1.update(set2)
                if not key1 in to_remove_set and not key2 in to_remove_set:
                    to_remove_set.add(key1)

    # Remove the overlapping keys outside of the loop
    for key in to_remove_set:
        number_sets.pop(key)

    # Intensity increase for each set
    intensity = int(255/(len(number_sets)+1))
    # Set intensity for each object
    for u in range(height):
        for v in range(width):
            i = 0
            for key in number_sets:
                i += 1
                if labels[u,v] in number_sets[key]: # Set intensity by looking up the sets
                    labels[u,v] = i*intensity
                    break

    return labels

''' Create and add to equivalence tables '''
def E_table(Lu, Ll, number_sets):

    create_new_set = False # Create new set flag

    if len(number_sets) == 0:
        number_sets[len(number_sets)+1] = set()    # If no set contains the label then create new
        number_sets[1].add(Lu)
        number_sets[1].add(Ll)
    else:
        for key in number_sets:
            if Lu in number_sets[key]:          # Check sets if label is contained then add to set
                number_sets[key].add(Lu)
                number_sets[key].add(Ll)
                create_new_set = False
            else:
                create_new_set = True

        if create_new_set == True:
            number_sets[len(number_sets)+1] = set()    # If no set contains the label then create new
            number_sets[len(number_sets)].add(Lu)
            number_sets[len(number_sets)].add(Ll)

    return number_sets

''' Filter by pixel count '''
def size_filter(labels, size_threshold): # Threshold can also be used
    # Get image dimensions
    height, width = labels.shape
    filter_out_label = []
    pixel_count_objects = []
    labels1 = labels.copy()
    unique_val = np.unique(labels1)

    # Get total pixel count for each object
    for i in unique_val:
        if i != 0:
            count = np.count_nonzero(labels1 == i)
            pixel_count_objects.append(count)

    # For Hand/Gun image hand has max pixels thus filter everything smaller than hand_pixel/2
    for i in unique_val:
        count = np.count_nonzero(labels1 == i)
        if count < max(pixel_count_objects)/2: # Use max(pixel_count_objects)/2 or size_threshold for filtering
            filter_out_label.append(i)

    # Set filtered out objects to black pixels
    for u in range(height):
        for v in range(width):
            if labels1[u,v] in filter_out_label:
                labels1[u,v] = 0

    return labels1

if __name__ == '__main__':
    main()
