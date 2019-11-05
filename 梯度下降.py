#梯度下降 f(x) = 3x^2 + 4x + 5
import numpy as np
import matplotlib.pyplot as plt  #函数集合

x_data=np.arange(-10,11).reshape([21,1]) #生成一个从-10到11的行向量，用reshape改成列向量
y_data=np.square(x_data)*3+x_data*4+5  #square是求平方

fig=plt.figure() #新建一个叫fig的绘图窗口
ax=fig.add_subplot(1,1,1)  #作为单个整数编码的子绘图网格参数。例如，“111”表示“1×1网格，第一子图”
ax.plot(x_data, y_data, lw=2)  #第一个网格开始画图 lw即line width
plt.ion()  #打开互动模式
plt.show()  #显示图像

start_x=10
step=0.1
current_x=start_x
current_y = 3 * current_x * current_x + 4 * current_x + 5
print("(loop_count, current_x, current_y)")
for i in range(10):
    print(i,current_x,current_y)
    derivative_f_x=6*current_x+4 #一阶导
    current_x=current_x-step*derivative_f_x
    current_y=3*current_x*current_x+4 * current_x + 5

    ax.scatter(current_x,current_y) #画散点图
    plt.pause(3) #停留的时间
