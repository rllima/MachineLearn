import pandas as pd
import numpy as np
import numpy.linalg as la

class lda(object):
    def __init__(self, dataset,target, class_values):
        self.data = dataset
        self.target = target
        self.size = self.data.shape[1] - 1
        self.class_values = class_values

    def calc_mean_vect(self):
        
        mean_vectors = []
        for value in self.class_values:
            mean_vec = self.data[self.data['defects'] == value].apply(lambda x : x.mean())
            mean_vectors.append([mean_vec[:-1], value])
        return mean_vectors
    
    def get_data_per_class(self):
        data_class = {}
        for cl in self.class_values:
            data_class[cl] = self.data[self.data['defects'] == cl].iloc[:,:-1]
        return data_class
    
    def calc_sw(self,mean_vectors,data_class):
        size = self.size
        sw = np.zeros((size,size))
        for cl, mv in zip(self.class_values,mean_vectors):
            class_sc_mat = np.zeros((size,size))
            data_cl_aux = data_class[cl]
            for values in data_cl_aux.values:
                values, mv[0] = np.array(values).reshape(size,1), np.array(mv[0]).reshape(size,1)
                x = values - mv[0]
                y = np.transpose(x)
                class_sc_mat += (x*y).astype(float)
            sw += class_sc_mat
        return sw
    
    def calc_sb(self, mean_vectors):
        size = self.size
        sb = np.zeros((size,size))
        overall_mean = np.mean(self.data.iloc[:,:-1], axis=0)
        for i,mean_vec in enumerate(mean_vectors):
            mean_vec[0] = np.array(mean_vec[0]).reshape(size,1)
            overall_mean = np.array(overall_mean).reshape(len(overall_mean),1)
            x = mean_vec[0] - overall_mean
            y = np.transpose(x)
            n = self.data[self.data['defects'] == mean_vec[1]].shape[0]
            sb += n * (x*y).astype(float)
        return sb
    
    def get_eigs(self,sw,sb):
        sw_inverse = la.inv(sw)
        eigen_val, eigen_vec = la.eig(sw_inverse.dot(sb))
        eig_pairs = [(np.abs(eigen_val[i]), eigen_vec[:,i]) for i in range(len(eigen_val))]
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    
    def get_EVR(self,eigen_val, eig_pairs):
        print('Variance explained:\n')
        eigv_sum = sum(eigen_val)
        for i,j in enumerate(eig_pairs):
            print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))
                
            
    