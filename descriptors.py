import cv2
import numpy as np

# Local dependencies
import utils
import constants

def orb(img):
    """
    Calculate the ORB descriptors for an image and resizes the image
    having the larger dimension set to 640 and keeping the size relation.

    Args:
        img (BGR matrix): The image that will be used.

    Returns:
        list of floats array: The descriptors found in the image.
    """
    orb = cv2.ORB()
    kp, des = orb.detectAndCompute(img, None)
    return des

def sift(img,img_descs,y,class_number):
    """
    Gets a list of 128 - dimensional descriptors using SIFT and DoG
    for keypoints and resizes the image having the larger dimension set to 640
    and keeping the size relation.

    Args:
        img (BGR matrix): The grayscale image that will be used.

    Returns:
        list of floats array: The descriptors found in the image.
    """
    
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    if des is not None:
        img_descs.append(des)
        y.append(class_number)
    else:
        print('Found you!!!!!!!')
    return img_descs,y

def descriptors_from_class(dataset, class_img_paths, class_number, option = constants.ORB_FEAT_OPTION):
    """
    Gets all the local descriptors for a class. If an image has a side with more than 640 pixels it will be resized
    leaving the biggest side at 640 pixels and conserving the aspect ratio for the other side.

    Args:
        dataset (Dataset object): An object that stores information about the dataset.
        class_img_paths (array of strings): The paths for each image in certain class.
        class_number (integer): The number of the class.
        option (integer): If this is 49 (The key '1') uses ORB features, else use SIFT.

    Returns:
        numpy float matrix: Each row are the descriptors found in an image of the class
    """
    des = None
    step = (constants.STEP_PERCENTAGE * len(class_img_paths)) / 100
    for i in range(len(class_img_paths)):
        img_path = class_img_paths[i]
        img = cv2.imread(img_path)
        resize_to = 640
        h, w, channels = img.shape
        if h > resize_to or w > resize_to:
            img = utils.resize(img, resize_to, h, w)
        if option == constants.ORB_FEAT_OPTION:
            des_name = "ORB"
            new_des = orb(img)
        else:
            des_name = "SIFT"
            new_des = sift(img)
        if new_des is not None:
            if des is None:
                des = np.array(new_des, dtype=np.float32)
            else:
                des = np.vstack((des, np.array(new_des)))
        # Print a message to show the status of the function
        if i % step == 0:
            percentage = (100 * i) / len(class_img_paths)
            message = "Calculated {0} descriptors for image {1} of {2}({3}%) of class number {4} ...".format(
                des_name, i, len(class_img_paths), percentage, class_number
            )
            print(message)
    message = "* Finished getting the descriptors for the class number {0}*".format(class_number)
    print(message)
    print("Number of descriptors in class: {0}".format(len(des)))
    print(class_number, len(des))
    dataset.set_class_count(class_number, len(des))
    return des

def all_descriptors(dataset, class_list, option = constants.ORB_FEAT_OPTION):
    """
    Gets every local descriptor of a set with different classes (This is useful for getting a codebook).

    Args:
        class_list (list of arrays of strings): The list has information for a specific class in each element and each
            element is an array of strings which are the paths for the image of that class.
        option (integer): It's 49 (the key '1') if ORB features are going to be used, else use SIFT features.

    Returns:
        numpy float matrix: Each row are the descriptors found in an image of the set
    """
    des = None
    for i in range(len(class_list)):
        message = "*** Getting descriptors for class number {0} of {1} ***".format(i, len(class_list))
        print(message)
        class_img_paths = class_list[i]
        new_des = descriptors_from_class(dataset, class_img_paths, i, option)
        if des is None:
            des = new_des
        else:
            des = np.vstack((des, new_des))
    message = "*****************************\n"\
              "Finished getting all the descriptors\n"
    print(message)
    print("Total number of descriptors: {0}".format(len(des)))
    if len(des) > 0:
        print("Dimension of descriptors: {0}".format(len(des[0])))
        print("First descriptor:\n{0}".format(des[0]))
    return des

def gen_codebook(dataset, descriptors, k = 64):
    """
    Generate a k codebook for the dataset.

    Args:
        dataset (Dataset object): An object that stores information about the dataset.
        descriptors (list of integer arrays): The descriptors for every class.
        k (integer): The number of clusters that are going to be calculated.

    Returns:
        list of integer arrays: The k codewords for the dataset.
    """
    k=int(k)
    print(type(dataset),type(descriptors),type(k))
    
    iterations = 10
    epsilon = 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, epsilon)
    compactness, labels, centers = cv2.kmeans(descriptors, k , None,criteria, iterations, cv2.KMEANS_RANDOM_CENTERS)
    return centers

def vlad(descriptors, centers):
    """
    Calculate the Vector of Locally Aggregated Descriptors (VLAD) which is a global descriptor from a group of
    descriptors and centers that are codewords of a codebook, obtained for example with K-Means.

    Args:
        descriptors (numpy float matrix): The local descriptors.
        centers (numpy float matrix): The centers are points representatives of the classes.

    Returns:
        numpy float array: The VLAD vector.
    """
    dimensions = len(descriptors[0])
    vlad_vector = np.zeros((len(centers), dimensions), dtype=np.float32)
    for descriptor in descriptors:
        nearest_center, center_idx = utils.find_nn(descriptor, centers)
        for i in range(dimensions):
            vlad_vector[center_idx][i] += (descriptor[i] - nearest_center[i])
    # L2 Normalization
    vlad_vector = cv2.normalize(vlad_vector,vlad_vector)
    vlad_vector = vlad_vector.flatten()
    return vlad_vector

def img_to_vect(img_descs, cluster_model):
    """
    Given an image path and a trained clustering model (eg KMeans),
    generates a feature vector representing that image.
    Useful for processing new images for a classifier prediction.
    """

    
    clustered_descs = [cluster_model.predict(raw_words) for raw_words in img_descs]
    img_bow_hist = np.array([np.bincount(clustered_desc, minlength=cluster_model.n_clusters) for clustered_desc in clustered_descs ])
    return img_bow_hist




def cluster_features(img_descs, cluster_model):
    """
    Cluster the training features using the cluster_model
    and convert each set of descriptors in img_descs
    to a Visual Bag of Words histogram.
    Parameters:
    -----------
    X : list of lists of SIFT descriptors (img_descs)
    training_idxs : array/list of integers
        Indicies for the training rows in img_descs
    cluster_model : clustering model (eg KMeans from scikit-learn)
        The model used to cluster the SIFT features
    Returns:
    --------
    X, cluster_model :
        X has K feature columns, each column corresponding to a visual word
        cluster_model has been fit to the training set
    """
    n_clusters = cluster_model.n_clusters
    # Concatenate all descriptors in the training set together
    training_descs = img_descs
    all_train_descriptors = [desc for desc_list in training_descs for desc in desc_list]
    all_train_descriptors = np.array(all_train_descriptors)

    if all_train_descriptors.shape[1] != 128:
        raise ValueError('Expected SIFT descriptors to have 128 features, got', all_train_descriptors.shape[1])

    # train kmeans or other cluster model on those descriptors selected above
    cluster_model.fit(all_train_descriptors)
    print('done clustering. Using clustering model to generate BoW histograms for each image.')

    # compute set of cluster-reduced words for each image
    img_clustered_words = [cluster_model.predict(raw_words) for raw_words in img_descs]

    # finally make a histogram of clustered word counts for each image. These are the final features.
    img_bow_hist = np.array(
        [np.bincount(clustered_words, minlength=n_clusters) for clustered_words in img_clustered_words])

    X = img_bow_hist
    print('done generating BoW histograms.')

    return X, cluster_model

