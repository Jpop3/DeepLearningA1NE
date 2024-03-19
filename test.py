import numpy as np

x = np.array([[1,2,3], [3,4,5], [6,7,8]])
y = np.array(([1,0], [0,1], [0,-1]))
p = np.random.permutation(len(x))
print(x[p], y[p])

print(sum(x)/len(x))