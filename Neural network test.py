import numpy as np
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
def dsigmoid(y):
    return y * (1 - y)
X = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 0],
              [0, 1, 1]])
y = np.array([[0],
            [1],
            [0],
            [1]])
np.random.seed(1)
w0 = 2*np.random.random((3,4)) - 1
w1 = 2*np.random.random((4,1)) - 1
print("x.shape=",X.shape)
print("y.shape=",y.shape)
print("w0.shape=",w0.shape)
print("w1.shape=",w1.shape)
print("++++++++++ start +++++++++++")
for i in range(200):
        # 三层网络
        L0 = X
        L1 = sigmoid(np.dot(L0, w0))  # 4行3列 * 3行4列 = 4行4列
        L2 = sigmoid(np.dot(L1, w1))  # 4行4列 * 4行1列 = 4行1列
        L2_loss = y - L2
        L2_delta = L2_loss * dsigmoid(L2)
        L1_loss = L2_delta.dot(w1.T)
        L1_delta = L1_loss * dsigmoid(L1)
        w1 += L1.T.dot(L2_delta)
        w0 += L0.T.dot(L1_delta)
        if (i % 20) == 0:
            print('L2=\n', L2)
            print("loss:" + str(np.mean(np.abs(L2_loss))))

