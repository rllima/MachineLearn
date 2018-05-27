import pca
import numpy as np
import pandas as pd

def main():
    columns_names = "loc,v(g),ev(g),iv(g),n,v,l,d,i,e,b,t,lOCode,lOComment,lOBlank,locCodeAndComment,uniq_Op,uniq_Opnd,total_Op,total_Opnd,branchCount,defects".split(',')
    df = pd.read_csv('List03\KC1.csv', names = columns_names)
    data = df.iloc[:,:-1].copy()
    target = df['defects']
    pca_instance = pca.pca(data,target)

    cov_matriz = pca_instance.cov_matriz()
    eigenvalues, eigenvectors = pca_instance.get_eigen_value_vector(cov_matriz)
    k_components = [1,3,5,9,15,20]
    for i in k_components:
        eigen_vec = pca_instance.get_eigenvecs(eigenvalues,eigenvectors,i)
        pca_instance.normalize()
        new_dataset = pca_instance.change_base(eigen_vec, pca_instance.normalize_data)
        print("Components:%.1d" % i)
        knns = [1,3,5]
        for j in knns:
            print("KNN = %.1d" % j)
            accuracy_pca = pca_instance.knn(new_dataset,j)
            accuracy_without_pca = pca_instance.knn(data,j)
            print("Acurracy with PCA:%.3f " % np.mean(accuracy_pca))
            print("Acurracy without PCA:%.3f\n" % np.mean(accuracy_without_pca))


if __name__ == '__main__':
	main()