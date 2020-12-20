from scipy.spatial.distance import euclidean
import pandas as pd
from sklearn import preprocessing
from fastdtw import fastdtw
import numpy as np

max = []
min = []
avg = []
mid = []
name = []
for u in range(1,41):
    for j in range(1,41):
        sum = 0
        d_list = []
        name.append('U'+str(u)+'s'+str(j))
        print('U'+str(u)+'s'+str(j))
        for s in range(1,21):
            if(j != s):
                file1 = '../svc2004_sizenormalized/Task2/U' + str(u) + 'S' + str(j) + '.TXT'
                file2 = '../svc2004_sizenormalized/Task2/U' + str(u) + 'S' + str(s) + '.TXT'
                s1 = pd.read_csv(file1, sep=",", header=None)
                #s1 = preprocessing.scale(s1)
                s2 = pd.read_csv(file2, sep=",", header=None)
                #s2 = preprocessing.scale(s2)
                distance, path = fastdtw(s1, s2, dist=euclidean)
                sum = sum + distance
                d_list.append(distance)
        max.append(np.max(d_list))
        min.append(np.min(d_list))
        avg.append(sum/len(d_list))
        mid.append(np.median(d_list))
dict = {'adict':name,'max':max,'min':min,'avg':avg,'mid':mid}
data = pd.DataFrame(dict)
data.reset_index()
print(data)
data.to_csv('task2_sizenormalized_feature_dtw.csv', index=False)

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