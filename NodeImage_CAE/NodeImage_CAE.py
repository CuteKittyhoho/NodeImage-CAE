#-*- coding:utf-8 -*-
import layers
import network
import creat_dataset
import save_vector
import save_model
import tensorflow as tf
class NodeImage_CAE:
    def __init__(self,file):
        self.dataset=creat_dataset.Dataset(file)            
        self.vector=save_vector.save_embedding_vector("F:/Dataset_for_CAE/",100)
        self.regularize_rate = 0.0001                       
        self.epoches=60                                    
        self.batch_size=20                                 
        self.image_size=0                                   
        self.image_chanel=1
        self.image_num=0                                    
        self.lost_function_mode=1                          
        self.optimizer_mode=0                               
        self.add_noise_to_dataset=False
        self.regularize_option=True
        
    def train(self):
        
        print("***************************\nStaring the whole process...\n***************************")
        self.image_size,self.image_num=self.dataset.get_shape()      
        inputs=tf.placeholder(tf.float32,[None,self.image_size,self.image_size, self.image_chanel], name='inputs') 
        #inputs=self.dataset.next_batch(self.batch_size)
        targets=tf.placeholder(tf.float32,[None,self.image_size,self.image_size, self.image_chanel], name='targets')
        print("**************************\ninputs shape is ",inputs.shape,"\n**************************")  
        if self.regularize_option: 
            regularizer = tf.contrib.layers.l2_regularizer(self.regularize_rate)
        else:            
            regularizer=None
        if self.add_noise_to_dataset:                                 
            inputs=layers.add_noise(inputs)       
        CAE_structure= network.structure(inputs,regularizer)        
        logit,encode=CAE_structure.forward()
        print("logit shape is " ,logit.shape)
        cost=layers.loss(logit,inputs,self.lost_function_mode)       
        if self.regularize_option:      
            cost=cost+tf.add_n(tf.get_collection('regularize_loss'))       
        optimizer=layers.Optimizer(cost,self.optimizer_mode,0.01)                                 
        print("***************************Staring to train model...***************************")
        with tf.Session() as sess:                      
            sess.run(tf.global_variables_initializer())                                          
            for epoch in range(self.epoches):                                                    
                sum_loss = 0
                n_batches=int(self.image_num / self.batch_size)+1                                                         
                print("***************************\nIn epoch %d,we have %d batches"%(epoch+1,n_batches))
                for i in range(n_batches):                                                                    
                    batch_x=self.dataset.next_batch(self.batch_size)                                      
                    batch_y=batch_x                                      
                    nonsense,loss_value,embedding=sess.run([optimizer,cost,encode],feed_dict={inputs:batch_x,targets:batch_y})                        
                    sum_loss = loss_value+sum_loss                    
                    if (epoch+1==self.epoches):                                                  
                        self.vector.append(embedding)                                           
                average_loss=sum_loss/n_batches
                print("In epoch %d/%d,average loss is %f"%(epoch+1,self.epoches, average_loss))
                if (epoch+1==self.epoches):
                    self.vector.save_as_mat()                                                                        
            print('Convolutional Autoencoder Training Process has Finished!')
            
def main():
    file="F:/Dataset_for_CAE/G_2.mat"
    CAE=NodeImage_CAE(file)
    CAE.train()

main()
