#-*- coding:utf-8 -*-
import tensorflow as tf



def conv2d(input,deep,size):
    conv=tf.layers.conv2d(input, deep, (size,size), padding='same', activation=tf.nn.relu)
    return conv



def max_pooling2d(input):
    maxpool=tf.layers.max_pooling2d(input, (2,2), (2,2), padding='same')
    return maxpool



def image_resize(input,reshape_size):
    res=tf.image.resize_nearest_neighbor(input,size=reshape_size )
    return res



def reshape(input):
    pool_shape = input.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    #print(pool_shape[0],pool_shape[1],pool_shape[2],pool_shape[3])
    reshaped = tf.reshape(input, [-1, nodes])
    return reshaped,nodes

def restore(input,res_len,res_wid,res_dep):
    restore_shape=[-1,res_len,res_wid,res_dep]
    return tf.reshape(input,restore_shape)
    


#----------------------------
def fully_connect(input,in_size, out_size,regularizer,scope_name):
    with tf.variable_scope(scope_name):
        weights = tf.get_variable("weight", 
                                     [in_size,out_size],
                                     #dtype=tf.float64,
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('regularize_loss', regularizer(weights))                   #将正则损失全部加入到集合losses中
        biases = tf.get_variable("bias", 
                                 [out_size], 
                                 #dtype=tf.float64,
                                 initializer=tf.constant_initializer(0.1))
        fc = tf.nn.relu(tf.matmul(input, weights))+biases
        return fc


#-------------------------------------------
def dropout(input, keep_rate):
    out = tf.nn.dropout(input, keep_rate)
    return out


#---------------------------------
def loss(input,target,mode):   
    if mode==0:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=input)      
        cost = tf.reduce_mean(loss)      
    if mode==1:
        loss=tf.square(tf.subtract(target, input))
        cost = tf.reduce_mean(loss)
    if mode==2:
        loss = tf.nn.l2_loss(input - target)  # L2 loss
        cost=loss
    return cost


#---------------------------------
def Optimizer(input,mode,learning_rate=None,globalstep=None):
    if mode==0:
        opt = tf.train.AdamOptimizer(0.001).minimize(input)           #AdmaOptimizerd的学习率是必须参数
    if mode==1:
        opt = tf.train.GradientDescentOptimizer(0.01).minimize(loss,golobal_step=globalstep)
    return opt


#---------------------------------
def add_noise(input_image_set):
    pass