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

    labels_face = seq_CCL(face_img)
    labels_gun = seq_CCL(gun_img)
    labels_test = seq_CCL(test_img)

    print(f"\n face num obs: {np.unique(labels_face)}")
    print(f"\n gun num obs: {np.unique(labels_gun)}")
    print(f"\n test num obs: {np.unique(labels_test)}")

    # Resize the image to 500x500 pixels
    # labels = cv2.resize(labels, (500, 500))

    cv2.imshow('Face CCL', np.uint8(labels_face))
    cv2.imshow('Gun CCL', np.uint8(labels_gun))
    cv2.imshow('Test CCL', np.uint8(labels_test))


    # Size filter on pixel amount
    labels_gun_filter = size_filter(labels_gun, 224)
    print(f"\n gun num obs after filter: {np.unique(labels_gun_filter)}")
    cv2.imshow('Gun CCL after size filter', np.uint8(labels_gun_filter))


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
    L_num = 0

    # Empty dictionary to hold all the sets/quivalence tables
    number_sets = {}
    # number_sets[1] = set()    # Create initial set
    num_set = 1 # TODO DELETE

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
                elif Lu != Ll and Lu > 0 and Ll > 0: # Both has labels and an item should be added to quivalence tables
                    labels[u,v] = min(Lu, Ll)
                    number_sets = E_table(Lu, Ll, number_sets, num_set)                  # Set these equal to each other in the quivalence tables
                    # if Lu == labels[u,v]:
                    #     Ll = Lu
                    # else:
                    #     Lu = Ll

                    # print("\n Both Not Zero")
                else:
                    L_num += 1
                    labels[u,v] = L_num
                    # print(L_num)
                    # print("\n New Cluster")

    # # Resize the image to 500x500 pixels
    # labels = cv2.resize(labels, (500, 500))

    # cv2.imshow('labels_prev', np.uint8(labels))

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

    # for key, value in number_sets.items():
    #     print(key, ':', value)

    intensity = int(255/(len(number_sets)+1)) # Intensity increase for each set

    for u in range(height):
        for v in range(width):
            i = 0
            for key in number_sets:
                i += 1
                if labels[u,v] in number_sets[key]: # Set intensity by looking up the sets
                    labels[u,v] = i*intensity
                    break

    return labels

# Create and add to quivalence tables
def E_table(Lu, Ll, number_sets, num_set):

    create_new_set = False

    if len(number_sets) == 0:
        number_sets[len(number_sets)+1] = set()    # If no set contains the label then create new
        # print("\n create new set")
        number_sets[1].add(Lu)
        number_sets[1].add(Ll)
    else:
        for key in number_sets:
            if Lu in number_sets[key]:          # Check sets if label is contained then add to set
                number_sets[key].add(Lu)
                number_sets[key].add(Ll)
                # print("\n add to set")
                create_new_set = False
            else:
                create_new_set = True

        if create_new_set == True:
            number_sets[len(number_sets)+1] = set()    # If no set contains the label then create new
            # print("\n create new set")
            number_sets[len(number_sets)].add(Lu)
            number_sets[len(number_sets)].add(Ll)

    return number_sets

def size_filter(labels, size_threshold):
    # Get image dimensions
    height, width = labels.shape
    filter_out_label = []
    pixel_count_objects = []

    unique_val = np.unique(labels)
    print(f"\n uni val = {unique_val}")
    for i in unique_val:
        if i != 0:
            count = np.count_nonzero(labels == i)
            print(f"\n count = {count}")
            pixel_count_objects.append(count)
            # if count < size_threshold:
            #     filter_out_label.append(i)

    print(f"\count array px = {max(pixel_count_objects)/2}")

    for i in unique_val:
        count = np.count_nonzero(labels == i)
        if count < max(pixel_count_objects)/2:
            filter_out_label.append(i)

    for u in range(height):
        for v in range(width):
            if labels[u,v] in filter_out_label:
                labels[u,v] = 0

    return labels



if __name__ == '__main__':
    main()