import cv2
import numpy as np
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

train_images_filenames = pickle.load(open('train_images_filenames.dat', 'rb'))
test_images_filenames = pickle.load(open('test_images_filenames.dat', 'rb'))
train_labels = pickle.load(open('train_labels.dat', 'rb'))
test_labels = pickle.load(open('test_labels.dat', 'rb'))

SIFTdetector = cv2.xfeatures2d.SIFT_create(nfeatures=300)

Train_descriptors = []
Train_label_per_descriptor = []

for filename, labels in zip(train_images_filenames, train_labels):
    ima = cv2.imread(filename)
    gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
    kpt, des = SIFTdetector.detectAndCompute(gray, None)
    Train_descriptors.append(des)
    Train_label_per_descriptor.append(labels)

D = np.vstack(Train_descriptors)

k = 128
codebook = MiniBatchKMeans(n_clusters=k, verbose=False, batch_size=k * 20, compute_labels=False,
                           reassignment_ratio=10 ** -4, random_state=42)
codebook.fit(D)

visual_words = np.zeros((len(Train_descriptors), k), dtype=np.float32)
for i in range(len(Train_descriptors)):
    words = codebook.predict(Train_descriptors[i])
    visual_words[i, :] = np.bincount(words, minlength=k)

metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
k_values = list(range(1, 21))
is_k_analisys = False
is_metric_analisys = True
results_accuracy = list()

if is_k_analisys:
    for kl in k_values:
        print(kl)
        knn = KNeighborsClassifier(n_neighbors=kl, n_jobs=-1, metric='minkowski')
        knn.fit(visual_words, train_labels)

        visual_words_test = np.zeros((len(test_images_filenames), k), dtype=np.float32)
        for i in range(len(test_images_filenames)):
            filename = test_images_filenames[i]
            ima = cv2.imread(filename)
            gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
            kpt, des = SIFTdetector.detectAndCompute(gray, None)
            words = codebook.predict(des)
            visual_words_test[i, :] = np.bincount(words, minlength=k)

        accuracy = 100 * knn.score(visual_words_test, test_labels)
        results_accuracy.append(accuracy)
    plt.plot(k_values, results_accuracy, 'go-', label='line 1', linewidth=2)
    plt.show()

if is_metric_analisys:
    for met in metrics:
        print(met)
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1, metric=met)
        knn.fit(visual_words, train_labels)



        visual_words_test = np.zeros((len(test_images_filenames), k), dtype=np.float32)
        for i in range(len(test_images_filenames)):
            filename = test_images_filenames[i]
            ima = cv2.imread(filename)
            gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
            kpt, des = SIFTdetector.detectAndCompute(gray, None)
            words = codebook.predict(des)
            visual_words_test[i, :] = np.bincount(words, minlength=k)

        accuracy = 100 * knn.score(visual_words_test, test_labels)
        results_accuracy.append(accuracy)
    plt.plot(metrics, results_accuracy, 'go-', label='line 1', linewidth=2)
    plt.show()

if not is_k_analisys and not is_metric_analisys:
    knn = KNeighborsClassifier(n_neighbors=19, n_jobs=-1, metric='minkowski')
    knn.fit(visual_words, train_labels)

    visual_words_test = np.zeros((len(test_images_filenames), k), dtype=np.float32)
    for i in range(len(test_images_filenames)):
        filename = test_images_filenames[i]
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        kpt, des = SIFTdetector.detectAndCompute(gray, None)
        words = codebook.predict(des)
        visual_words_test[i, :] = np.bincount(words, minlength=k)

    accuracy = 100 * knn.score(visual_words_test, test_labels)
    print(accuracy)