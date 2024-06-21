import cv2
import numpy as np
import time
from sklearn.svm import LinearSVC 
from sklearn.externals import joblib
from sklearn.cluster import MiniBatchKMeans
# Local dependencies

import constants
import descriptors
import filenames
import utils


class Classifier:
    """
    Class for making training and testing in image classification.
    """
    def __init__(self, dataset, log):
        """
        Initialize the classifier object.
        Args:
            dataset (Dataset): The object that stores the information about the dataset.
            log (Log): The object that stores the information about the times and the results of the process.

        Returns:
            void
        """
        self.dataset = dataset
        self.log = log

    def train(self, svm_kernel, k, des_name, des_option=constants.ORB_FEAT_OPTION, is_interactive=True):
        """
        Gets the descriptors for the training set and then calculates the SVM for them.

        Args:
            svm_kernel (constant): The kernel of the SVM that will be created.
            codebook (NumPy float matrix): Each row is a center of a codebook of Bag of Words approach.
            des_option (integer): The option of the feature that is going to be used as local descriptor.
            is_interactive (boolean): If it is the user can choose to load files or generate.

        Returns:
            cv2.SVM: The Support Vector Machine obtained in the training phase.
        """
        isTrain= True
        des_name = constants.ORB_FEAT_NAME if des_option == constants.ORB_FEAT_OPTION else constants.SIFT_FEAT_NAME
        x_filename = filenames.vlads_train(k, des_name)
        print("Getting global descriptors for the training set.")
        start = time.time()
        x, y, cluster_model = self.get_data_and_labels(self.dataset.get_train_set(),None, k, des_name ,des_option,isTrain)
        utils.save(x_filename, x)
        end = time.time()
        svm_filename = filenames.svm(k, des_name, svm_kernel)
        print("Calculating the Support Vector Machine for the training set...")
        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(svm_kernel)
        svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
        svm.train(x, cv2.ml.ROW_SAMPLE, y)
        return svm, cluster_model

    def test(self, svm, cluster_model, k, des_option = constants.ORB_FEAT_OPTION, is_interactive=True):
        """
        Gets the descriptors for the testing set and use the svm given as a parameter to predict all the elements

        Args:
            codebook (NumPy matrix): Each row is a center of a codebook of Bag of Words approach.
            svm (cv2.SVM): The Support Vector Machine obtained in the training phase.
            des_option (integer): The option of the feature that is going to be used as local descriptor.
            is_interactive (boolean): If it is the user can choose to load files or generate.

        Returns:
            NumPy float array: The result of the predictions made.
            NumPy float array: The real labels for the testing set.
        """
        isTrain = False
        des_name = constants.ORB_FEAT_NAME if des_option == constants.ORB_FEAT_OPTION else constants.SIFT_FEAT_NAME
        print("Getting global descriptors for the testing set...")
        start = time.time()
        x, y, cluster_model= self.get_data_and_labels(self.dataset.get_test_set(), cluster_model, k, des_name,isTrain,des_option)
        end = time.time()
        start = time.time()
        _, result = svm.predict(x)
        end = time.time()
        self.log.predict_time(end - start)
        mask = result == y
        correct = np.count_nonzero(mask)
        accuracy = (correct * 100.0 / result.size)
        self.log.accuracy(accuracy)
        return result, y

    def get_data_and_labels(self, img_set, cluster_model, k, des_name, codebook,isTrain, des_option = constants.ORB_FEAT_OPTION):
        """
        Calculates all the local descriptors for an image set and then uses a codebook to calculate the VLAD global
        descriptor for each image and stores the label with the class of the image.
        Args:
            img_set (string array): The list of image paths for the set.
            codebook (numpy float matrix): Each row is a center and each column is a dimension of the centers.
            des_option (integer): The option of the feature that is going to be used as local descriptor.

        Returns:
            NumPy float matrix: Each row is the global descriptor of an image and each column is a dimension.
            NumPy float array: Each element is the number of the class for the corresponding image.
        """
        y = []
        x = None
        img_descs = []
        
        for class_number in range(len(img_set)):
            img_paths = img_set[class_number]
            
            step = round(constants.STEP_PERCENTAGE * len(img_paths) / 100)
            for i in range(len(img_paths)):
                if (step > 0) and (i % step == 0):
                    percentage = (100 * i) / len(img_paths)
                img = cv2.imread(img_paths[i])
                
                des,y = descriptors.sift(img,img_descs,y,class_number)
        isTrain = int(isTrain)
        if isTrain == 1:
            X, cluster_model = descriptors.cluster_features(des,cluster_model=MiniBatchKMeans(n_clusters=64))
        else:
            X = descriptors.img_to_vect(des,cluster_model)
        print('X',X.shape,X)
        y = np.int32(y)[:,np.newaxis]
        x = np.matrix(X, dtype=np.float32)
        return x, y, cluster_model