import cv2

# Local dependencies
import constants


def codebook(k, des_name):
    print('files.dirname')
    print(constants.FILES_DIR_NAME)
    return "{0}/codebook_{1}.mat".format(constants.FILES_DIR_NAME, signature(k, des_name))


def result(k, des_name, svm_kernel):
    return "{0}/result_{1}.csv".format(constants.FILES_DIR_NAME, signature(k, des_name, kernel_name(svm_kernel)))


def vlads_train(k, des_name):
    return "{0}/VLADS_train_{1}.mat".format(constants.FILES_DIR_NAME, signature(k, des_name))


def vlads_test(k, des_name):
    return "{0}/VLADS_test_{1}.mat".format(constants.FILES_DIR_NAME, signature(k, des_name))


def svm(k, des_name, svm_kernel):
    return "{0}/svm_data_{1}.dat".format(constants.FILES_DIR_NAME, signature(k, des_name, kernel_name(svm_kernel)))


def log(k, des_name, svm_kernel):
    return "{0}/log_{1}.txt".format(constants.FILES_DIR_NAME, signature(k, des_name, kernel_name(svm_kernel)))


def signature(k, des_name, svm_kernel=None):
    if svm_kernel is None:
        return "{0}_{1}".format(k, des_name)
    else:
        return "{0}_{1}_{2}".format(k, des_name, svm_kernel)

def kernel_name(svm_kernel):
    if svm_kernel == cv2.ml.SVM_LINEAR:
        kernel_name = "LINEAR"
    elif svm_kernel == cv2.ml.SVM_POLY:
        kernel_name = "POLY"
    elif svm_kernel == cv2.ml.SVM_RBF:
        kernel_name = "RBF"
    else:
        kernel_name = "SIGMOID"
    return kernel_name
