
# Local dependencies
import utils
import filenames


class Log:
    def __init__(self, k, des_name, svm_kernel):
        self.text = ""
        self.k = k
        self.des_name = des_name
        self.svm_kernel = svm_kernel

    def save(self):
        file = open(filenames.log(self.k, self.des_name, self.svm_kernel), "w")
        file.write(self.text)
        file.close()

    def train_des_time(self, time):
        str = "Time for getting all the local descriptors of the training images was {0}.\n"
        elapsed_time = utils.humanize_time(time)
        self.text += str.format(elapsed_time)

    def codebook_time(self, time):
        str = "Time for generating the codebook with k-means was {0}.\n"
        elapsed_time = utils.humanize_time(time)
        self.text += str.format(elapsed_time)

    def train_vlad_time(self, time):
        self.vlad_time(time, "training")

    def svm_time(self, time):
        str = "Time for calculating the SVM was {0}.\n"
        elapsed_time = utils.humanize_time(time)
        self.text += str.format(elapsed_time)

    def test_vlad_time(self, time):
        self.vlad_time(time, "testing")

    def predict_time(self, time):
        elapsed_time = utils.humanize_time(time)
        self.text += "Elapsed time predicting the testing set is {0}\n".format(elapsed_time)

    def accuracy(self, accuracy):
        self.text += "Accuracy = {0}.\n".format(accuracy)

    def classes(self, classes):
        self.text += "Classes = {0}\n".format(classes)

    def classes_counts(self, classes_counts):
        self.text += "Classes Local Descriptors Counts = {0}\n".format(classes_counts)

    def confusion_matrix(self, conf_mat):
        self.text += "Confusion Matrix =\n{0}".format(conf_mat)

    def vlad_time(self, time, set):
        str = "Time for getting VLAD global descriptors of the {0} images was {1}.\n"
        elapsed_time = utils.humanize_time(time)
        self.text += str.format(set, elapsed_time)