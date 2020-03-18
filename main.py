import cv2
import joblib
import numpy as np
import time
import os

# Local dependencies
from classifier import Classifier
from dataset import Dataset
import descriptors
import constants
import utils
import filenames
from log import Log


def main(is_interactive=True, k=64, des_option=constants.ORB_FEAT_OPTION, svm_kernel=cv2.ml.SVM_LINEAR):
    if not is_interactive:
        experiment_start = time.time()
    # Check for the dataset of images
    if not os.path.exists(constants.DATASET_PATH):
        print("Dataset not found, please copy one.")
        return
    dataset = Dataset(constants.DATASET_PATH)
    dataset.generate_sets()

    # Check for the directory where stores generated files
    if not os.path.exists(constants.FILES_DIR_NAME):
        os.makedirs(constants.FILES_DIR_NAME)

    if is_interactive:
        des_option = int(input("Enter [1] for using ORB features or [2] to use SIFT features.\n"))
        k = input("Enter the number of cluster centers you want for the codebook.\n")
        svm_option = int(input("Enter [1] for using SVM kernel Linear or [2] to use RBF.\n"))
        svm_kernel = cv2.ml.SVM_LINEAR if svm_option == 1 else cv2.ml.SVM_RBF

    des_name = constants.ORB_FEAT_NAME if des_option == constants.ORB_FEAT_OPTION else constants.SIFT_FEAT_NAME
    print(des_name)
    log = Log(k, des_name, svm_kernel)
    

    codebook_filename = filenames.codebook(k, des_name)
    print('codebook_filename')
    print(codebook_filename)
    start = time.time()   
    end = time.time()
    log.train_des_time(end - start)
    start = time.time()
    end = time.time()
    log.codebook_time(end - start)
    # Train and test the dataset
    classifier = Classifier(dataset, log)
    svm, cluster_model = classifier.train(svm_kernel, k, des_name, des_option=des_option, is_interactive=is_interactive)
    svm.save("svm_result.dat")
    joblib.dump(cluster_model, 'cluster_model.plk')
    print("Training ready. Now beginning with testing")
    result, labels = classifier.test(svm, cluster_model, k, des_option=des_option, is_interactive=is_interactive)
    print('test result')
    print(result,labels)
    # Store the results from the test
    classes = dataset.get_classes()
    log.classes(classes)
    log.classes_counts(dataset.get_classes_counts())
    result_filename = filenames.result(k, des_name, svm_kernel)
    test_count = len(dataset.get_test_set()[0])
    result_matrix = np.reshape(result, (len(classes), test_count))
    utils.save_csv(result_filename, result_matrix)

    # Create a confusion matrix
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=np.uint32)
    for i in range(len(result)):
        predicted_id = int(result[i])
        real_id = int(labels[i])
        confusion_matrix[real_id][predicted_id] += 1

    print("Confusion Matrix =\n{0}".format(confusion_matrix))
    log.confusion_matrix(confusion_matrix)
    log.save()
    print("Log saved on {0}.".format(filenames.log(k, des_name, svm_kernel)))
    if not is_interactive:
        experiment_end = time.time()
        elapsed_time = utils.humanize_time(experiment_end - experiment_start)
        print("Total time during the experiment was {0}".format(elapsed_time))
    else:
        # Show a plot of the confusion matrix on interactive mode
        utils.show_conf_mat(confusion_matrix)
        #raw_input("Press [Enter] to exit ...")

if __name__ == '__main__':
    main()