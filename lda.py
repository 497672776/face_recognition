# coding=utf-8
import numpy as np
import os
import cv2
from sklearn import neighbors, model_selection
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def get_label_from_file(file_path):
    label_num_dict = {'male': 0, 'female': 1}
    labels_np = np.array([])
    with open(file_path) as f:
        for each_line in f:
            begin = each_line.find('_sex ')
            end = each_line.find(')', begin)
            if begin == -1:
                continue
            sex = each_line[begin + 6:end]
            labels_np = np.append(labels_np, label_num_dict[sex])
            # label_index = re.sub(r'\D', "", each_line)
            # print(label_index)
        f.close()
    return labels_np


def create_data_base(path):
    file_list = os.listdir(path)
    data_np = []
    for file in file_list:
        input_file_name = r'F:\Course\PR\input\face_recognition\face\rawdata'
        input_file_name = input_file_name + '/' + file
        nx = 128
        ny = 128
        file_load = np.fromfile(input_file_name, dtype='u1')
        if file_load.shape[0] != nx * ny:
            file_load = np.reshape(file_load, (512, 512))
            file_load = cv2.resize(file_load, (128, 128))
        file_load = np.reshape(file_load, (nx * ny,))
        # print(file)
        data_np.append(file_load)
    data_np = np.array(data_np)
    return data_np


def draw_chart(dimension, accuracy, line_color):
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.figure(figsize=(6,6))
    plt.plot(dimension, accuracy, line_color)
    plt.xlabel('lda降维的维度')
    plt.ylabel('人脸识别准确率')
    plt.title('对比不同核函数训练结果准确性的差异 ')
    plt.savefig("result.png")
    plt.show()


def face_fuc(data_train, labels_train, data_test, label_test, k, c):
    dimension = [1]
    accuracy = []
    for i in dimension:
        lda =LDA(n_components=i).fit(data_train,labels_train)
        data_numpy_train_lda = lda.transform(data_train)
        data_numpy_test_lda = lda.transform(data_test)

        # data standard
        stdScaler = StandardScaler().fit(data_numpy_train_lda)
        data_numpy_train_lda = stdScaler.transform(data_numpy_train_lda)
        data_numpy_test_lda = stdScaler.transform(data_numpy_test_lda)

        svm = SVC(kernel=k).fit(data_numpy_train_lda, labels_train)
        face_target_pred = svm.predict(data_numpy_test_lda)
        true = 0
        for i in range(0,len(face_target_pred)):
            if face_target_pred[i] == label_test[i]:
                true += 1
        accuracy.append(true / label_test.shape[0])
    print(accuracy)
    plt.plot(dimension, accuracy, c)


# get data use lda
data_path = r'F:\Course\PR\input\face_recognition\face\rawdata'
data_numpy = create_data_base(data_path)

# get label
file_name_DR = r'F:\Course\PR\input\face_recognition\face\faceDR'
file_name_DS = r'F:\Course\PR\input\face_recognition\face\faceDS'
DR_numpy = get_label_from_file(file_name_DR)
DS_numpy = get_label_from_file(file_name_DS)
labels_numpy = np.append(DR_numpy, DS_numpy)


# get the min class num
each_class_num = min(len(data_numpy[labels_numpy == 0]), len(data_numpy[labels_numpy == 1]))

# quantity_proportion
quantity_proportion = [0.9, 0.1]
# get dataset
train_index = int(quantity_proportion[0]*each_class_num)

data_numpy_train = np.vstack((data_numpy[labels_numpy == 0][0:train_index],
                              data_numpy[labels_numpy == 1][0:train_index]))
data_numpy_test = np.vstack((data_numpy[labels_numpy == 0][train_index:each_class_num],
                             data_numpy[labels_numpy == 1][train_index:each_class_num]))
labels_numpy_train = np.hstack((labels_numpy[labels_numpy == 0][0:train_index],
                                labels_numpy[labels_numpy == 1][0:train_index]))
labels_numpy_test = np.hstack((labels_numpy[labels_numpy == 0][train_index:each_class_num],
                               labels_numpy[labels_numpy == 1][train_index:each_class_num]))

# data standard
stdScaler = StandardScaler().fit(data_numpy_train)
data_numpy_train = stdScaler.transform(data_numpy_train)
data_numpy_test = stdScaler.transform(data_numpy_test)

lda = LDA(n_components=1).fit(data_numpy_train, labels_numpy_train)
data_numpy_train_lda = lda.transform(data_numpy_train)
data_numpy_test_lda = lda.transform(data_numpy_test)

# data standard
stdScaler = StandardScaler().fit(data_numpy_train_lda)
data_numpy_train_lda = stdScaler.transform(data_numpy_train_lda)
data_numpy_test_lda = stdScaler.transform(data_numpy_test_lda)

data_numpy = np.vstack((data_numpy_train_lda, data_numpy_test_lda))
labels_numpy = np.hstack((labels_numpy_train, labels_numpy_test))

n_neighbors_list = [1, 3, 5, 7, 9, 11, 13, 15, 17]
accuracy_mean_list = []
accuracy_std_list = []

for k in n_neighbors_list:
    print('--------------------------------')
    print("n_neighbors = ", str(k))
    # knn classification
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)

    # K-fold validation
    scores = cross_val_score(knn, data_numpy, labels_numpy, cv=10)
    print("Through K-fold validation, we know the accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    accuracy_mean_list.append(scores.mean())
    accuracy_std_list.append(scores.std())
    # F1
    scores = cross_val_score(knn, data_numpy, labels_numpy, cv=10, scoring='f1')
    print('F1: ', scores)

    # confusion matrix
    y_pred = knn.fit(data_numpy_train_lda, labels_numpy_train).predict(data_numpy_test_lda)
    print('confusion matrix: ')
    print(confusion_matrix(labels_numpy_test, y_pred))

    # ROC/AUC
    print('roc_auc: ')
    scores = cross_val_score(knn, data_numpy, labels_numpy, cv=10, scoring='roc_auc')
    print(scores)

# example data


plt.rcParams['font.sans-serif'] = 'SimHei'
plt.figure(figsize=(6, 6))
plt.xlabel('The n_neighbors of KNN')
plt.ylabel('The accuracy')
plt.errorbar(n_neighbors_list, accuracy_mean_list, accuracy_std_list, fmt='-o')
plt.savefig("knn_result.png")
plt.show()

