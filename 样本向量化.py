import numpy as np  #加载numpy数组库

a1=np.random.standard_normal((5,3))  #随机生成两个5*3的二维数组
a2=np.random.standard_normal((5,3))

print(a1)
print(3*a1+1)

a3=np.random.standard_normal(3)  #表示生成一个标准正态分布的一维数组a3，a3的长度为3
print(a3)
print(a1+a3)  #a1+a3表示对a1使用a3广播

print(a1.transpose())  #a1的转置

def f(x):
    return 2*x
print(f(a1))  #f(a1)表示当自变量为数组a1时，对应的结果（数组a1的每个元素都作用于函数f）