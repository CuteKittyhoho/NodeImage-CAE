#-*- coding:utf-8 -*-
import numpy as np
import scipy.io as sio
class save_embedding_vector:
    def __init__(self,save_file_path,vector_len):
        self.file_path=save_file_path
        self.embedding=np.random.rand(0,vector_len)
        
    def append(self,tensor):                  
        self.embedding=np.vstack((self.embedding,tensor))
        return self.embedding
       
    def save_as_npy(self):
        np.save(self.file_path,self.temp)
        
    def save_as_mat(self):
        sio.savemat(self.file_path + '/embedding.mat',{'embedding':self.embedding})
    
    def save_as_txt(self):
        pass