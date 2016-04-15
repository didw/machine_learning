import numpy as np
import matplotlib.pyplot as plt

def sigmoid(a):
  return 1/(1+np.exp(-a))

def costFunction(W, X, y):
  m = len(y)
  J = np.dot(-y.transpose(), np.log(sigmoid(np.dot(X,W)))) - np.dot((1 - y).transpose(), np.log(1 - sigmoid(np.dot(X,W))));
  J = J / m;
  grad = np.dot(X.transpose(), (sigmoid(np.dot(X,W)) - y)) / m;
  return J, grad


def main():
  with open('ex2data2.txt') as f:
    lines = f.read().splitlines()

  m = len(lines)
  words = np.array(map(lambda x: x.split(','), lines), np.float).reshape(m, 3)

  X = words[:,0:2].reshape(m,2)
  y = words[:,2].reshape(m,1)


  pos = (y==1); neg = (y==0);

  plt.figure()
  plt.savefig('foo.png')
  plt.plot(X(pos,1), X(pos,2), 'k+', 'linewidth', 2, 'markersize', 7)
  plt.plot(X(neg,1), X(neg,2), 'ko', 'markerfacecolor', 'y', 'markersize', 7)


  W = np.zeros((3,1))
  X = np.append(np.ones(m), X.transpose()).reshape(3,m).transpose()


  lr = 0.1
  for i in range(1000):
    J, dW = costFunction(W, X, y)
    if i % 100 == 0:
      print(i, 'th loss: ', J)
    W = W - lr*dW
  pred = np.array(np.dot(X,W) + 0.5, np.int)
  pred = sigmoid(np.dot(X,W))
  print(np.sum(y>=0.5)*100/m)
  print(np.sum((pred >= 0.5) == y)*100/m)



if __name__ == '__main__':
  main()


