import numpy as np


def MDS(D, d):
    """
    :param D: 原始距离矩阵
    :param d: 缩放后维数
    :return: 返回缩放后的距离矩阵
    """

    D = np.asarray(D)
    DSquare = D ** 2
    totalMean = np.mean(DSquare)
    columnMean = np.mean(DSquare, axis=0)
    rowMean = np.mean(DSquare, axis=1)
    B = np.zeros(DSquare.shape)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i][j] = -0.5 * (DSquare[i][j] - rowMean[i] - columnMean[j] + totalMean)
    eigVal, eigVec = np.linalg.eig(B)  # 求特征值及特征向量
    # 对特征值进行排序，得到排序索引
    eigValSorted_indices = np.argsort(eigVal)
    # 提取d个最大特征向量
    topd_eigVec = eigVec[:, eigValSorted_indices[:-d - 1:-1]]  # -d-1前加:才能向左切
    X = np.dot(topd_eigVec, np.sqrt(np.diag(eigVal[:-d - 1:-1])))
    return X


D = [[0, 587, 1212, 701, 1936, 604, 748, 2139, 2182, 543],
     [587, 0, 920, 940, 1745, 1188, 713, 1858, 1737, 597],
     [1212, 920, 0, 879, 831, 1726, 1631, 949, 1021, 1494],
     [701, 940, 879, 0, 1374, 968, 1420, 1645, 1891, 1220],
     [1936, 1745, 831, 1374, 0, 2339, 2451, 347, 959, 2300],
     [604, 1188, 1726, 968, 2339, 0, 1092, 2594, 2734, 923],
     [748, 713, 1631, 1420, 2451, 1092, 0, 2571, 2408, 205],
     [2139, 1858, 949, 1645, 347, 2594, 2571, 0, 678, 2442],
     [2182, 1737, 1021, 1891, 959, 2734, 2408, 678, 0, 2329],
     [543, 597, 1494, 1220, 2300, 923, 205, 2442, 2329, 0]]
X = MDS(D, 3)
print(X)
