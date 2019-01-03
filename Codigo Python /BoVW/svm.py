import cv2
import numpy as np
import pickle
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import time
from sklearn import metrics
from sklearn import svm
import utils
import seaborn as sns;

sns.set()
from sklearn.model_selection import GridSearchCV

sampleClassificationReport = "{0} {1} {2} {3}".format('precision', 'recall', 'f1-score', 'norm') + '\n'
train_images_filenames = pickle.load(open('train_images_filenames.dat', 'rb'))
test_images_filenames = pickle.load(open('test_images_filenames.dat', 'rb'))
train_labels = pickle.load(open('train_labels.dat', 'rb'))
test_labels = pickle.load(open('test_labels.dat', 'rb'))
verbose_analysis = True

if verbose_analysis:

    crossValidationValues = [2, 3, 5, 10, 15, 30, 50]
    Cs = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    gammas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma': gammas}

    for crossVal in crossValidationValues:
        print(crossVal)
        sampleClassificationReportCustom = "{0} {1} {2} {3}".format('precision', 'recall', 'f1-score',
                                                                    'Arguments Kernel') + '\n'
        SVMKernels = ['linear', 'rbf', 'poly2', 'poly3', 'poly4', 'poly5', 'sigmoid']
        accuracies = list()
        precisions = list()
        recalls = list()
        f1s = list()
        time_list = list()

        for kernel in SVMKernels:
            print(kernel)
            step_size = 15
            start = time.time()
            SIFTdetector = cv2.xfeatures2d.SIFT_create()
            Train_descriptors = []
            Train_label_per_descriptor = []
            for filename, labels in zip(train_images_filenames, train_labels):
                ima = cv2.imread(filename)
                x, y = ima.shape[:2]
                h = int(x)
                w = int(y)
                res = cv2.resize(ima, dsize=(h, w))
                gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size)
                      for x in range(0, gray.shape[1], step_size)]
                kpt, des = SIFTdetector.compute(gray, kp)
                Train_descriptors.append(des)
                Train_label_per_descriptor.append(labels)
            D = np.vstack(Train_descriptors)
            k = 140
            codebook = MiniBatchKMeans(n_clusters=k, verbose=False, batch_size=k * 20, compute_labels=False,
                                       reassignment_ratio=10 ** -4, random_state=42)
            codebook.fit(D)
            visual_words = np.zeros((len(Train_descriptors), k), dtype=np.float32)
            for i in range(len(Train_descriptors)):
                words = codebook.predict(Train_descriptors[i])
                visual_words[i, :] = np.bincount(words, minlength=k)

            if kernel == 'poly2':
                sv = svm.SVC(kernel='poly', degree=2)
            elif kernel == 'poly3':
                sv = svm.SVC(kernel='poly', degree=3)
            elif kernel == 'poly4':
                sv = svm.SVC(kernel='poly', degree=4)
            elif kernel == 'poly5':
                sv = svm.SVC(kernel='poly', degree=5)
            else:
                sv = svm.SVC(kernel=kernel)

            grid_search = GridSearchCV(sv, param_grid, cv=crossVal)
            grid_search.fit(visual_words, train_labels)

            print(grid_search.best_params_)
            print(grid_search.cv_results_)

            visual_words_test = np.zeros((len(test_images_filenames), k), dtype=np.float32)
            results = list()
            for i in range(len(test_images_filenames)):
                filename = test_images_filenames[i]
                ima = cv2.imread(filename)
                x, y = ima.shape[:2]
                h = int(x)
                w = int(y)
                res = cv2.resize(ima, dsize=(h, w))
                gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size)
                      for x in range(0, gray.shape[1], step_size)]
                kpt, des = SIFTdetector.compute(gray, kp)
                words = codebook.predict(des)
                visual_words_test[i, :] = np.bincount(words, minlength=k)

            predict_labels = grid_search.predict(visual_words_test)

            utils.plot_data(visual_words, train_labels, visual_words_test, predict_labels)

            accuracy = metrics.accuracy_score(test_labels, predict_labels)
            precision = metrics.precision_score(test_labels, predict_labels, average='macro')
            recall = metrics.recall_score(test_labels, predict_labels, average='macro')
            f1 = metrics.f1_score(test_labels, predict_labels, average='macro')

            end = time.time()
            tTime = end - start

            print('Accuracy' + str(accuracy))
            print('Precision' + str(precision))
            print('Recall' + str(recall))
            print('F1 Score' + str(f1))
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            time_list.append(tTime)
            sampleClassificationReportCustom += ("{0} {1} {2} {3} {4}".format(kernel, precision, recall, f1,
                                                                              str(grid_search.best_params_).replace(" ",
                                                                                                                    "")) + '\n')

            mat = metrics.confusion_matrix(test_labels, predict_labels)
            sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                        xticklabels=np.unique(predict_labels),
                        yticklabels=np.unique(predict_labels))
            plt.xlabel('true label')
            plt.ylabel('predicted label')
            plt.show()

            report = metrics.classification_report(test_labels, predict_labels, target_names=np.unique(predict_labels))
            utils.plot_classification_report(report)

            print('For Kernel=' + str(kernel) + ' the accuracy is: ' + str(accuracy))

        plt.plot(SVMKernels, accuracies, 'go-', label='Accuracy', linewidth=2, color='blue')
        plt.plot(SVMKernels, time_list, 'go-', label='Time', linewidth=2, color='green')
        plt.legend(['Accuracy', 'Time'])
        plt.xlabel('Kernel type')
        plt.title('SVM Kernels')
        plt.show()

        plt.plot(SVMKernels, precisions, 'go-', label='Precision', linewidth=2, color='blue')
        plt.plot(SVMKernels, recalls, 'go-', label='Recall', linewidth=2, color='red')
        plt.plot(SVMKernels, f1s, 'go-', label='F1', linewidth=2, color='green')
        plt.legend(['Precision', 'Recall', 'F1'])
        plt.xlabel('Kernels')
        plt.title('SVM Kernels')
        plt.ylim([0, 1])
        plt.show()

        utils.plot_classification_report_custom(sampleClassificationReportCustom,
                                                'Analysis Kernel SVM with cross validation ' + str(crossVal))
        plt.savefig('test_plot_classif_report.png', dpi=200, format='png', bbox_inches='tight')
        plt.close()
