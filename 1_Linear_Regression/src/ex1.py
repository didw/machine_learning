import numpy as np

with open('ex1data1.txt') as f:
  lines = f.read().splitlines()

data = np.array(map(lambda x: x.split(','), lines), np.float)
X = data[:,0]
y = data[:,1].reshape(97,1)

m = len(X)

X = np.append(np.ones(m), X).reshape(2,97).transpose()
W = np.zeros((2,1))

lr = 0.01

for _ in range(10000):
  W -= lr / m * np.dot((np.dot(X,W) - y).transpose(), X).transpose()

print('loss: ', np.mean((np.dot(X,W) - y)**2))
