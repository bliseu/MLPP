# -*- coding: utf-8 -*-
#!/usr/bin/python3

import tensorflow as tf
import numpy as np

a2=np.loadtxt('input1.txt')#训练的输入数据
a1=[139,1,139,139,1,139,1,1,12,12,12,2,2,12,123,123,38,1,12,12]
a3=[-4.7262,-4.7261,-4.7252,-4.7249,-4.7246,-4.7240,-4.7232,-4.7204,-4.7184,
       -4.7164,-4.7154,-4.7151,-4.7113,-4.7076,-4.6887,-4.6881,-4.6876,-4.6868,
       -4.6822,-4.6818]
a1= np.reshape(a1,(20,1))
a3= np.reshape(a3,(20,1))
myin=np.hstack((a2,a1,a3))
myout=np.loadtxt('input2.txt')#训练的输出数据
myout=np.reshape(myout,(20,9))
myout=(myout-0.5)*100
def make_layer(inputs, in_size, out_size, activate=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    basis = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    result = tf.matmul(inputs, weights) + basis
    #tf.summary.scalar('wx_b', result) 
    if activate is None:
        return result
    else:
        return activate(result)


class BPNetwork:
    def __init__(self, session):
        self.session = session
        self.loss = None
        self.optimizer = None
        self.input_n = 0
        self.hidden_n = 0
        self.hidden_size = []
        self.output_n = 0
        self.input_layer = None
        self.hidden_layers = []
        self.output_layer = None
        self.label_layer = None

    def setup(self, layers):
        # set size args
        if len(layers) < 3:
            return
        self.input_n = layers[0]
        self.hidden_n = len(layers) - 2  #隐藏层数
        self.hidden_size = layers[1:-1]  #每个隐藏层节点数
        self.output_n = layers[-1]
        #建立网络
        self.input_layer = tf.placeholder(tf.float32, [None, self.input_n])
        self.label_layer = tf.placeholder(tf.float32, [None, self.output_n])
        #建立隐藏层
        in_size = self.input_n
        out_size = self.hidden_size[0]
        self.hidden_layers.append(make_layer(self.input_layer, in_size, out_size, activate=tf.nn.relu))
        for i in range(self.hidden_n-1):
            in_size = out_size
            out_size = self.hidden_size[i+1]
            inputs = self.hidden_layers[-1]
            self.hidden_layers.append(make_layer(inputs, in_size, out_size, activate=tf.nn.relu))
        #建立输出层
        self.output_layer = make_layer(self.hidden_layers[-1], self.hidden_size[-1], self.output_n)

    def train(self, cases, labels, limit=500, learn_rate=0.2):
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.reduce_sum(tf.square((self.label_layer - self.output_layer)), reduction_indices=[1]))
        #tf.summary.histogram('loss',loss) #可视化观看变量
        #tf.summary.scalar('loss',loss) #可视化观看常量  
        optimizer = tf.train.GradientDescentOptimizer(learn_rate)
        with tf.name_scope('train'):
            train = optimizer.minimize(loss)
        init=tf.global_variables_initializer()
        #merged = tf.summary.merge_all() 
        #writer = tf.summary.FileWriter("/Users/user/.spyder-py3/log",self.session.graph) 
        self.session.run(init)
        
        for i in range(limit):
            self.session.run(train, feed_dict={self.input_layer: cases, self.label_layer: labels})
#            if i % 10== 0: 
#                ddd = self.session.run(merged,feed_dict={self.input_layer: cases, self.label_layer: labels})  
#                writer.add_summary(ddd, i) 
         
              
    def predict(self, case):
        return self.session.run(self.output_layer, feed_dict={self.input_layer: case})

    def test(self):    
        x_data = myin[1:]
        y_data = myout[1:]
        test_data = myin[0:1]
        self.setup([8,9,9])
        self.train(x_data, y_data)
        oout=self.predict(test_data)
        oout=oout/30+0.5
        print(np.reshape(oout,(3,3)))

def main():
    with tf.Session() as session:
        model = BPNetwork(session)
        model.test()


if __name__ == '__main__':
    main()