import cv2
import numpy as np
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import time
from sklearn import metrics

verbose_analysis = True

train_images_filenames = pickle.load(open('train_images_filenames.dat','rb'))
test_images_filenames = pickle.load(open('test_images_filenames.dat','rb'))
train_labels = pickle.load(open('train_labels.dat','rb'))
test_labels = pickle.load(open('test_labels.dat','rb'))


if verbose_analysis:
    step_sizes = [5,7,10,15,20,25,30,35,40,50,75,100]
    nfeatures_list = [950]
    accuracies = list()
    time_list = list()

    for step_size in step_sizes:
        start = time.time()
        print(step_size)
        SIFTdetector = cv2.xfeatures2d.SIFT_create()

        Train_descriptors = []
        Train_label_per_descriptor = []
        for filename, labels in zip(train_images_filenames, train_labels):
            ima = cv2.imread(filename)
            gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
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

        knn = KNeighborsClassifier(n_neighbors=19, n_jobs=-1, metric='euclidean')
        knn.fit(visual_words, train_labels)

        visual_words_test = np.zeros((len(test_images_filenames), k), dtype=np.float32)
        for i in range(len(test_images_filenames)):
            filename = test_images_filenames[i]
            ima = cv2.imread(filename)
            gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
            kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size)
                  for x in range(0, gray.shape[1], step_size)]
            kpt, des = SIFTdetector.compute(gray, kp)
            words = codebook.predict(des)
            visual_words_test[i, :] = np.bincount(words, minlength=k)

        accuracy = 100*knn.score(visual_words_test, test_labels)

        end = time.time()
        tTime = end - start
        accuracies.append(accuracy)
        time_list.append(tTime)
        print('For nfeatures=' + str(step_size) + ' the accuracy is: ' + str(accuracy))
    plt.plot(step_sizes, accuracies, 'go-', label='Accuracy', linewidth=2, color='blue')
    plt.plot(step_sizes, time_list, 'go-', label='Time', linewidth=2,color='green')
    plt.legend(['Accuracy','Time'])
    plt.xlabel('Steps')
    plt.show()


