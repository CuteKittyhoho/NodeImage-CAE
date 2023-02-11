#-*- coding:utf-8 -*-
import h5py
import numpy as np
class Dataset:                 
    def __init__(self, path):
        self.raw_data=h5py.File(path,'r')
        self.keys=self.raw_data.keys()
        self.dataset=np.array(self.raw_data['images'])
        self.data_point=0
        
    def get_shape(self):
        shape=self.dataset.shape
        #print("dataset shape is" ,shape)
        size=shape[1]
        num=shape[0]
        return size,num

    def get_type(self):
        data_type=type(self.dataset)
        return data_type

    def next_batch(self,batch_size):
        shape=self.dataset.shape                                
        image_num=shape[0]                                      
        image_size=shape[1]
        data=self.dataset
        start=self.data_point%image_num                         
        end=min(self.data_point+batch_size,image_num)

        if self.data_point+batch_size<image_num:
           self.data_point=self.data_point+batch_size  
        else:                                                                  
            self.data_point=0

        raw_batch=data[start:end,:,:]
        batch=np.reshape(raw_batch,(raw_batch.shape[0],raw_batch.shape[1],raw_batch.shape[2],1))    
        return batch                                                                                                                             
