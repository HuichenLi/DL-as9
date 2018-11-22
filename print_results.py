import numpy as np
from helperFunctions import getUCF101
NUM_CLASSES = 101


def print_confusion_matrix(class_list, file_path='single_frame_confusion_matrix.npy'):
    confusion_matrix = np.load(file_path)

    results = np.diag(confusion_matrix)
    indices = np.argsort(results)

    sorted_list = np.asarray(class_list)
    sorted_list = sorted_list[indices]
    sorted_results = results[indices]

    for i in range(NUM_CLASSES):
        # print(sorted_list[i], sorted_results[i], number_of_examples[indices[i]])
        print("%s & %.4f \\\\" %(sorted_list[i], sorted_results[i]))


def main():
    data_directory = '/projects/training/bauh/AR/'
    class_list, train, test = getUCF101(base_directory=data_directory)
    print_confusion_matrix(class_list=class_list)


if __name__ == '__main__':
    main()