import tensorflow as tf
import numpy
from tensorflow.examples.tutorials.minist import input_data

mnist=input_data.read_data_sets('./MNIST',one_hot=True)

#input_Layer
x = tf.placeholder(dtype=tf.float32,shape=[None ,784],name='x')   #28*28*1=一张图片
#Label
y = tf.placeholder(dtype=tf.float32,shape=[None ,10],name='y')

batch_size = 1000

def add_layer(input_data,input_num,output_num,activation_function=None):
    #ouput = input_data * weight + bias
    w = tf.variable(initial_value=tf.random_normal(shape=[input_num,output_num]))
    b = tf.variable(initial_value=tf.random_normal(shape=[1, output_num]))
    output = tf.add(tf.matmul(input_data,w),b)
    if activation_function:
        output = activation_function(output)
    return output


def build_nn(data):
    hidden_layer1 = add_layer(data,784,100,activation_function=tf.nn.sigmoid)
    hidden_layer2 = add_layer(hidden_layer1,100,50,activation_function=tf.nn.sigmoid)
    output_layer = add_layer(hidden_layer2,50,10)
    return output_layer


def train_nn(data):
    #output of NN
    output = build_nn(data)

    #这个函数的功能就是计算labels和logits之间的交叉熵
    loss = tf.reduce_mean (tf.softmax_cross_with_logits(label=y,logits=output))
    optimizer=tf.train .GradientDescentOptimizer(learning_rate=1).minimize(loss)   #梯度下降优化器

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  #对所有数据进行初始化

        for i in range(50):
             epoch_cost=0
             for _ in range(int(mnist.train.num_examples / batch_size)):
                 x_data,y_data = mnist.train.next_batch(batch_size)  #取一批数据进来
                 cost,_ =sess.run([loss,optimizer],feed_dict={x:x_data,y:y_data})
                 epoch_cost += cost
             print('Epoch',i,':',epoch_cost)
        accuracy = tf.equal_mean(tf.cast(tf.argmax(y,1),tf.equal(tf.argmax(output,1)),tf.float32))
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print(acc)






train_nn()