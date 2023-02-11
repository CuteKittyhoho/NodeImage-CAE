#-*- coding:utf-8 -*-
import tensorflow as tf
import layers
class structure:
    def __init__(self,input_tensor,regularizer=None):
        self.input=input_tensor              
        self.CONV1_DEEP = 8
        self.CONV1_SIZE = 3        
        self.CONV2_DEEP = 16
        self.CONV2_SIZE = 3        
        self. FC1_SIZE = 100
        self. FC2_SIZE = 5776             
        self.REG=regularizer        
        self.KEEP_RATE=0.9

    def forward(self):
        conv1 = layers.conv2d(self.input, self.CONV1_DEEP, self.CONV1_SIZE)
        print("conv1 shape is",conv1.shape)                  
        maxpool1 = layers.max_pooling2d(conv1)
        print("maxpool1 shape is",maxpool1.shape)           
        conv2 = layers.conv2d(maxpool1, self.CONV2_DEEP, self.CONV2_SIZE)
        print("conv2 shape is",conv2.shape)                
        maxpool2 = layers.max_pooling2d(conv2)
        print("maxpool2 shape is",maxpool2.shape)                              
        reshaped_tensor,nodes=layers.reshape(maxpool2)
        print("reshaped_tensor shape is",reshaped_tensor.shape)                 
        fc1=layers.fully_connect(reshaped_tensor,nodes,self.FC1_SIZE,self.REG,"fc1")
        print("fc1 shape is",fc1.shape)       
        fc2=layers.fully_connect(fc1,self.FC1_SIZE,self.FC2_SIZE,self.REG,"encoded")
        print("fc2 shape is",fc2.shape)              
        restored_tensor=layers.restore(fc2,19,19,16)
        print("restored_tensor shape is",restored_tensor.shape)                  
        upsample1 = layers.image_resize(restored_tensor, (37,37))
        print("upsample1 shape is",upsample1.shape)               
        conv3 = layers.conv2d(upsample1, self.CONV1_DEEP, self.CONV1_SIZE)
        print("conv3 shape is",conv3.shape)                      
        upsample2 = layers.image_resize(conv3, (73,73))
        print("upsample2 shape is",upsample2.shape)               
        conv4 = layers.conv2d(upsample2, self.CONV1_DEEP, self.CONV1_SIZE,)
        print("conv4 shape is",conv4.shape)                       
        logits = tf.layers.conv2d(conv4, 1, (3,3), padding='same', activation=None)              
        print("logits shape is",logits.shape)
        return logits,fc1
