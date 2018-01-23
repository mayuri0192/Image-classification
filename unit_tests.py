import cv2
import time
import pickle
import numpy as np
from matplotlib import pyplot as plt

#----------------------------------
# Local dependencies
import descriptors
import utils
from dataset import Dataset
import constants
import filenames

def test_dataset():
    dataset = Dataset(constants.DATASET_PATH)
    pickle.dump(dataset, open(constants.DATASET_OBJ_FILENAME, "wb"), protocol=constants.PICKLE_PROTOCOL)
    classes = dataset.get_classes()
    print("Dataset generated with {0} classes.".format(len(classes)))
    print(classes)
    train = dataset.get_train_set()
    test = dataset.get_test_set()
    for i in range(len(classes)):
        print(
            "There are {0} training files and {1} testing files for class number {2} ({3})".format(
                len(train[i]), len(test[i]), i, classes[i]
            )
        )

def test_des_type():
    img = cv2.imread(constants.TESTING_IMG_PATH)
    kp, des = descriptors.orb(img)
    return des

def test_descriptors():
    img = cv2.imread(constants.TESTING_IMG_PATH)
    cv2.imshow("Normal Image", img)
    print("Normal Image")
    option = input("Enter [1] for using ORB features and other number to use SIFT.\n")
    start = time.time()
    if option == 1:
        orb = cv2.ORB()
        kp, des = orb.detectAndCompute(img, None)
    else:
        sift = cv2.SIFT()
        kp, des = sift.detectAndCompute(img, None)
    end = time.time()
    elapsed_time = utils.humanize_time(end - start)
    des_name = constants.ORB_FEAT_NAME if option == ord(constants.ORB_FEAT_OPTION_KEY) else constants.SIFT_FEAT_NAME
    print("Elapsed time getting descriptors {0}".format(elapsed_time))
    print("Number of descriptors found {0}".format(len(des)))
    if des is not None and len(des) > 0:
        print("Dimension of descriptors {0}".format(len(des[0])))
    print("Name of descriptors used is {0}".format(des_name))
    img2 = cv2.drawKeypoints(img, kp)
    # plt.imshow(img2), plt.show()
    cv2.imshow("{0} descriptors".format(des_name), img2)
    print("Press any key to exit ...")
    cv2.waitKey()

def test_codebook():
    dataset = pickle.load(open(constants.DATASET_OBJ_FILENAME, "rb"))
    option = input("Enter [1] for using ORB features or [2] to use SIFT features.\n")
    start = time.time()
    des = descriptors.all_descriptors(dataset, dataset.get_train_set(), option)
    end = time.time()
    elapsed_time = utils.humanize_time(end - start)
    print("Elapsed time getting all the descriptors is {0}".format(elapsed_time))
    k = 64
    des_name = constants.ORB_FEAT_NAME if option == constants.ORB_FEAT_OPTION else constants.SIFT_FEAT_NAME
    codebook_filename = "codebook_{0}_{1}.csv".format(k, des_name)
    start = time.time()
    codebook = descriptors.gen_codebook(dataset, des, k)
    end = time.time()
    elapsed_time = utils.humanize_time(end - start)
    print("Elapsed time calculating the k means for the codebook is {0}".format(elapsed_time))
    np.savetxt(codebook_filename, codebook, delimiter=constants.NUMPY_DELIMITER)
    print("Codebook loaded in {0}, press any key to exit ...".format(constants.CODEBOOK_FILE_NAME))
    cv2.waitKey()

def test_vlad():
    img = cv2.imread(constants.TESTING_IMG_PATH)
    option = input("Enter [1] for using ORB features or [2] to use SIFT features.\n")
    if option == 1:
        des = descriptors.orb(img)
    else:
        des = descriptors.sift(img)
    des_name = constants.ORB_FEAT_NAME if option == constants.ORB_FEAT_OPTION else constants.SIFT_FEAT_NAME
    k = 128
    codebook_filename = "codebook_{0}_{1}.csv".format(k, des_name)
    centers = np.loadtxt(codebook_filename, delimiter=constants.NUMPY_DELIMITER)
    vlad_vector = descriptors.vlad(des, centers)
    print(vlad_vector)
    return vlad_vector

def test_one_img_classification():
    img = cv2.imread("test.jpg")
    resize_to = 640
    h, w, channels = img.shape
    img = utils.resize(img, resize_to, h, w)
    des = descriptors.sift(img)
    k = 128
    des_name = "SIFT"
    codebook_filename = filenames.codebook(k, des_name)
    codebook = utils.load(codebook_filename)
    img_vlad = descriptors.vlad(des, codebook)
    svm_filename = filenames.svm(k, des_name)
    svm = cv2.SVM()
    svm.load(svm_filename)
    result = svm.predict(img_vlad)
    print("result is {0}".format(result))

if __name__ == '__main__':
    test_descriptors()