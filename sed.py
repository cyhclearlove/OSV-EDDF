# ! /usr/bin/python
# -*- coding: utf8 -*-
import pandas as pd
from sklearn import preprocessing
import numpy as np
from MyProject.signature import Signature
#计算SED距离
t1 = 2
t2 = 4
def distance(p1,p2):
    d = 0
    for i in range(0,len(p1)):
        d = d + abs(p1[i] - p2[i])
    return d

def Levenshtein_Distance(str1, str2):
    """
    计算字符串 str1 和 str2 的编辑距离
    :param str1
    :param str2
    :return:
    """
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    # print(len(matrix),len(matrix[0]))

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            d = distance(str1[i-1],str2[j-1])
            if d < t1:
                matrix[i][j] = min(matrix[i - 1][j] + 0.5, matrix[i][j - 1] + 0.5, matrix[i - 1][j - 1])
            else:
                matrix[i][j] = min(matrix[i - 1][j], matrix[i][j - 1], matrix[i - 1][j - 1]) + d*d
    return matrix[len(str1)][len(str2)]

max_ = []
min_ = []
avg_ = []
mid_ = []
name = []
# x，y，时间，按压状态，压力，水平，垂直，angle，v_x，v_y，v，curva，a
dtype = {0: float,1:float,2:float,3:float,4:float,5:float,6:float,7:float,8:float,9:float,10:float,11:float,12:float}
for u in range(1,101):
    for j in range(1,51):
        sum = 0
        d_list = []
        name.append('U'+str(u)+'s'+str(j))
        print(u,j)
        file1 = '../mcyt_more_feature/U' + str(u) + 'S' + str(j) + '.txt'
        s1 = pd.read_csv(file1, sep=",", header=None,dtype=dtype)
        #s1 = s1.drop([2,3],1)
        s1 = preprocessing.scale(s1)
        for s in range(26,31):
            if(j != s):
                file2 = '../mcyt_more_feature/U' + str(u) + 'S' + str(s) + '.txt'
                s2 = pd.read_csv(file2, sep=",", header=None,dtype=dtype)
                #s2 = s2.drop([2,3],1)
                s2 = preprocessing.scale(s2)
                distance_ = Levenshtein_Distance(s1, s2)
                sum = sum + distance_
                d_list.append(distance_)
        max_.append(np.max(d_list))
        min_.append(np.min(d_list))
        avg_.append(sum/len(d_list))
        mid_.append(np.median(d_list))
dict = {'adict':name,'sed_max':max_,'sed_min':min_,'sed_avg':avg_,'sed_mid':mid_}
data = pd.DataFrame(dict)
data.reset_index()
print(data)
data.to_csv('5_mcyt_feature_sed.csv', index=False)

#打标签
# lable = []
# data = pd.read_csv('feature_dtw.csv', header=0)
# j = 0
# for i in range(1,len(data)+1):
#     if j<20:
#         lable.append(1)
#     else:
#         lable.append(0)
#     j = j+1
#     j = j%40
# print(lable)
# data['y_lable'] = lable
# data.to_csv('feature_dtw.csv')