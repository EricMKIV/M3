import cv2
import numpy as np
import pickle
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import time
from sklearn import metrics
from sklearn import svm
import seaborn as sns; sns.set()
import sklearn.preprocessing as pre

sampleClassificationReport = "{0} {1} {2} {3}".format('precision', 'recall', 'f1-score', 'norm') + '\n'
train_images_filenames = pickle.load(open('train_images_filenames.dat', 'rb'))
test_images_filenames = pickle.load(open('test_images_filenames.dat', 'rb'))
train_labels = pickle.load(open('train_labels.dat', 'rb'))
test_labels = pickle.load(open('test_labels.dat', 'rb'))
verbose_analysis = True

if verbose_analysis:

    preprocessing = ['wo', 'stdSc', 'l2']
    #preprocessing = ['wo', 'stdSc', 'l2']
    methods = ['Without preprocessing', 'Standard Scalar', 'L2 norm']

    accuracies = list()
    precisions = list()
    recalls = list()
    f1s = list()
    time_list = list()

    for method in preprocessing:

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

            if method == 'stdSc':
                desf = pre.StandardScaler().fit(des)
                des = desf.transform(des)
            elif method == 'l2':
                des = pre.normalize(des, norm='l2')

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

        sv = svm.SVC(kernel='poly', degree=3, gamma=0.01, C=0.0001)
        sv.fit(visual_words, train_labels)

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

            if method == 'stdSc':
                desf = pre.StandardScaler().fit(des)
                des = desf.transform(des)
            elif method == 'l2':
                des = pre.normalize(des, norm='l2')

            words = codebook.predict(des)
            visual_words_test[i, :] = np.bincount(words, minlength=k)

        predict_labels = sv.predict(visual_words_test)

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

        accuracies.append(accuracy * 100)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        time_list.append(tTime)

        mat = metrics.confusion_matrix(test_labels, predict_labels)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                    xticklabels=np.unique(predict_labels),
                    yticklabels=np.unique(predict_labels))
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        plt.show()

    plt.plot(methods, accuracies, 'go-', label='Accuracy', linewidth=2, color='blue')
    plt.plot(methods, time_list, 'go-', label='Time', linewidth=2, color='green')
    plt.legend(['Accuracy', 'Time'])
    plt.xlabel('Reduction')
    plt.title('Reduction Scale')
    plt.show()

    plt.plot(methods, precisions, 'go-', label='Accuracy', linewidth=2, color='blue')
    plt.plot(methods, recalls, 'go-', label='Accuracy', linewidth=2, color='red')
    plt.plot(methods, f1s, 'go-', label='Time', linewidth=2, color='green')
    plt.legend(['Precision', 'Recall', 'F1'])
    plt.xlabel('Methods')
    plt.title('Methods Preprocessing Features')
    plt.ylim([0, 1])
    plt.show()