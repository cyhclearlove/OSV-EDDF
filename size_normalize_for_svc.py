import pandas as pd
import numpy as np
from sklearn import preprocessing
for u in range(1,41):
    for s in range(1,41):
        print('U' + str(u) + 'S' + str(s) )
        file1 = '../svc2004/Task1/U' + str(u) + 'S' + str(s) + '.TXT'
        data = pd.read_csv(file1, sep=" ", header=None)
        # 保持长宽比
        mean_x = np.mean(data[0])
        mean_y = np.mean(data[1])
        std_x = 50
        std_y = 50
        for i in range(0, len(data)):
            data[0].values[i] = (data[0].values[i] - mean_x) / std_x
            data[1].values[i] = (data[1].values[i] - mean_y) / std_y
        data.to_csv('../svc2004_sizenormalized/Task1/U' + str(u) + 'S' + str(s) + '.TXT', index=False, header=None)