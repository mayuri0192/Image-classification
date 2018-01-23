import numpy.random as nprnd
import numpy as np
import cv2
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py 


def random_split(l, sample_size):
    """
    Randomly splits a list in two parts. A sample and a rest (other part).

    Args:
        l (list): A list that is going to be splitted.
        sample_size (integer): The size of the sample that is going to be taken.

    Returns:
        list: One random group from the list.
        list: Another group from the list with size equal to sample_size.
    """
    sample_indices = nprnd.choice(len(l), size=sample_size, replace=False)
    # print (len(sample_indices))
    sample_indices.sort()
    # print("sample_indices = {0}".format(sample_indices))
    other_part = []
    sample_part = []
    indices_counter = 0
    for index in range(len(l)):
        current_elem = l[index]
        if indices_counter == sample_size:
            other_part = other_part + l[index:]
            break
        if index == sample_indices[indices_counter]:
            sample_part.append(current_elem)
            indices_counter += 1
        else:
            other_part.append(current_elem)
    return other_part, sample_part

def humanize_time(secs):
    """
    Extracted from http://testingreflections.com/node/6534
    """
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    return '%02d:%02d:%02f' % (hours, mins, secs)

def resize(img, new_size, h, w):
    """
    Changes the largest side of an image to the new size and changes the other to maintain the aspect ratio.

    Args:
        img (BGR Matrix): The image that is going to be resized.
        new_size (integer): The value wanted for the biggest side of the image.

    Returns:
        BGR Matrix: The image resized to the new value keeping the aspect ratio.
    """
    if h > w:
        new_h = 640
        new_w = (640 * w) / h
    else:
        new_h = (640 * h) / w
        new_w = 640
    img = cv2.resize(img, (new_w, new_h))
    return img

def find_nn(point, neighborhood):
    """
    Finds the nearest neighborhood of a vector.

    Args:
        point (float array): The initial point.
        neighborhood (numpy float matrix): The points that are around the initial point.

    Returns:
        float array: The point that is the nearest neighbor of the initial point.
        integer: Index of the nearest neighbor inside the neighborhood list
    """
    min_dist = float('inf')
    nn = neighborhood[0]
    nn_idx = 0
    for i in range(len(neighborhood)):
        neighbor = neighborhood[i]
        dist = cv2.norm(point - neighbor)
        if dist < min_dist:
            min_dist = dist
            nn = neighbor
            nn_idx = i

    return nn, nn_idx

def save(filename, arr):
    """
    Stores a numpy array in a file.

    Args:
        filename (string): The name for the file.
        arr (numpy array):

    Returns:
        void
    """
    data = {"stored": arr}
    sio.savemat(filename, data)

def load(filename):
    """
    Load a numpy array from a file.

    Args:
        filename (string): The name of the file.

    Returns:
        numpy array: The array stored in the file.
    """
    #data = sio.loadmat(filename)
    
    
    f = h5py.File(filename,'r') 
    data = f.get('data/variable1') 
    data = np.array(data) # For converting to numpy array
    return data["stored"]

def save_csv(filename, arr):
    """
    Save an array into a file with the filename.

    Args:
        filename (string): The name for the file.
        arr (numpy array): The array that is going to be saved.

    Returns:
        void
    """
    file = open(filename, "w")
    for row in arr:
        for i in range(len(row) - 1):
            file.write("{0} ".format(row[i]))
        file.write("{0}\n".format(row[len(row) - 1]))

def show_conf_mat(confusion_matrix):
    """
    Show a windows with a color image for a confusion matrix

    Args:
        confusion_matrix (NumPy Array): The matrix to be shown.

    Returns:
        void
    """
    plt.matshow(confusion_matrix)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.show()