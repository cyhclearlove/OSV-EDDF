# coding=utf-8
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']# 汉语显示
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

#sig = pd.read_csv('task2_sizenormalized_feature.csv', header=0)
sig = pd.read_csv('task1_feature.csv', header=0)

sig['sed_avg'] = preprocessing.scale(sig['sed_avg'])
sig['sed_max'] = preprocessing.scale(sig['sed_max'])
sig['sed_mid'] = preprocessing.scale(sig['sed_mid'])
sig['sed_min'] = preprocessing.scale(sig['sed_min'])
sig['dtw_avg'] = preprocessing.scale(sig['dtw_avg'])
sig['dtw_max'] = preprocessing.scale(sig['dtw_max'])
sig['dtw_mid'] = preprocessing.scale(sig['dtw_mid'])
sig['dtw_min'] = preprocessing.scale(sig['dtw_min'])


data = np.array(sig.drop(['adict'], axis=1))


l_far = []
l_frr = []
l_thresholds=[]
result = []
l_oor = []
for i in range(1,41):
    print('loop::::',i)
    X = data[40*(i-1):40*i,0:8]
    y = data[40*(i-1):40*i,8]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=0)
    print("train::::",y_train)
    print('test::::',y_test)
    X_train = x_train
    Y_train = y_train
    clf = svm.SVC(kernel='linear', probability=True)
    #clf = KNeighborsClassifier(n_neighbors=2)
    #clf = RandomForestClassifier()
    #clf = GaussianNB()
    #clf.fit(X_train, Y_train)
    #y_scores = cross_val_score(clf, X_train, Y_train, cv=8)#交叉验证
    #y_scores = cross_val_predict(clf, X_train, Y_train, cv=4, method="decision_function")
    clf.fit(X_train,Y_train)
    y_scores = clf.decision_function(x_test)
    #y_scores = clf.predict(x_test)
    #print(y_scores)
    # score = roc_auc_score(y_test, y_scores)
    # result.append(score)
    print('y_scores::::::',y_scores)
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    # print('fpr::',fpr)
    # print('tpr::',tpr)
    far = fpr
    frr = 1- tpr
    print('far::',far)
    print('frr::',frr)
    orr = 1 - np.min(far+frr)/2
    l_oor.append(orr)
    index = np.where((far + frr) == min(far + frr))
    l_far.append(round(far[index[0][0]],3))
    l_frr.append(round(frr[index[0][0]],3))
    l_thresholds.append(round(thresholds[index[0][0]],3))
print(len(l_oor),len(l_far),len(l_frr),len(l_thresholds))
print('l_thresholds:::',l_thresholds)
# print('平均auc：',np.average(result))
# print('最大auc：',np.max(result))
# print('最小auc：',np.min(result))
print('平均orr：',np.average(l_oor))
# print('最大orr：',np.max(l_oor))
# print('最小orr：',np.min(l_oor))
# print(l_far)
# print(l_frr)
print('平均far:',np.average(l_far))
print('平均frr:',np.average(l_frr))

plt.plot(range(0,40), l_far,'*-', linewidth=3,label='FAR')
plt.plot(range(0,40), l_frr, 'o-',linewidth=1,label='FRR')
plt.legend()
plt.show()
# plt.plot(range(0,41), l_oor, linewidth=2, label=label)
# plt.plot(range(1,41), l_thresholds, '*-', linewidth=2)
# plt.xlabel('用户')
# plt.ylabel('阈值')
# plt.show()
# fpr, tpr, thresholds = roc_curve(Y_train, y_scores)
# plot_roc_curve(fpr, tpr)
# plt.show()