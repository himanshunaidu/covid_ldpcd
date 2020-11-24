#SVM to classify images based on all the features given by the networks (standard and LDP)
import numpy as np
from sklearn import svm
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import confusion_matrix
import time
import random

from mysql_feat_extract import features, mydb, mycursor, tupleToList
from check_runtime import checkRuntime
from config import coronet_train_table, coronet_test_table

start_time = time.time()
dense_train_info, dense_train_features = features(mydb, mycursor, coronet_train_table)
dense_train_class = tupleToList(dense_train_info)
std_scale = StandardScaler().fit(dense_train_features)
std_train_features = std_scale.transform(dense_train_features)

#Check Time
train_time = checkRuntime("Run Train Feature Extraction", start_time)

clf = svm.SVC(kernel='rbf', degree=3, gamma='scale', coef0=random.uniform(0, 1)) #, probability=True)
clf.fit(std_train_features, dense_train_class)

#Check Time
svm_time = checkRuntime("Run SVC", train_time)

dense_test_info, dense_test_features = features(mydb, mycursor, coronet_test_table)
dense_test_class = tupleToList(dense_test_info)
std_test_features = std_scale.transform(dense_test_features)

predict_test_class = clf.predict(std_test_features)
print(predict_test_class)
cm = confusion_matrix(dense_test_class, predict_test_class)
print(cm)

#Check Time
predict_time = checkRuntime("Run SVC", svm_time)