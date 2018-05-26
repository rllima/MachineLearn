import numpy as np
import pandas as pd
from sklearn import preprocessing
import numpy.linalg as la
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class pca(object):
    def __init__(self,dataset,target):
        self.data = dataset
        self.normalize_data = []
        self.target = target
        
    def normalize(self):
        min_max_scaler = preprocessing.MinMaxScaler()
        np_scaled = min_max_scaler.fit_transform(self.data)
        df_normalized = pd.DataFrame(np_scaled)
        self.normalize_data = df_normalized

    def cov_matriz(self):
        return np.cov(np.transpose(self.data))

    def get_eigen_value_vector(self,cov_matriz):
        return la.eig(cov_matriz)

    def get_eigenvecs(self,eigenvalues, eigenvectors, k_eigenvecs):
        #Indices dos dos autovalores em ordem decrescente. É necessário manter os indices após ordenação, para poder recuperar os autovetores correspondentes
        indexes = np.argsort(eigenvalues)[::-1]
        eigenvectors_trans = np.transpose(eigenvectors)
        eigen_vec = []
        for i in range(k_eigenvecs):
            eigen_vec.append(eigenvectors_trans[indexes[i]])
        return eigen_vec
            
    def get_EVR(self,eigenvalue):
        EVR=eigenvalue/np.sum(eigenvalue)
        return EVR

    def change_base(self,eigen_vec, data_normalized):
        trans = []
        for index, row in data_normalized.iterrows():
            trans.append(np.dot(eigen_vec, row))
        return pd.DataFrame(trans)

    def knn(self,dataset):
        skf = StratifiedKFold(n_splits=3)
        accuracy = []
        for train, test in skf.split(dataset,self.target):
            data_train, data_test = dataset.iloc[train], dataset.iloc[test]
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(data_train, self.target.iloc[train])
            pred = knn.predict(data_test)
            accuracy.append(accuracy_score(self.target.iloc[test], pred))
        return accuracy



