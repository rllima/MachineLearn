import pca
import numpy as np
import pandas as pd

def main():
    columns_names = "loc,v(g),ev(g),iv(g),n,v,l,d,i,e,b,t,lOCode,lOComment,lOBlank,locCodeAndComment,uniq_Op,uniq_Opnd,total_Op,total_Opnd,branchCount,defects".split(',')
    df = pd.read_csv('F:\AM\Listas\MachineLearn\List03\KC1.csv', names = columns_names)
    data = df.iloc[:,:-1].copy()
    target = df['defects']
    pca_instance = pca.pca(data,target)

    cov_matriz = pca_instance.cov_matriz()
    eigenvalues, eigenvectors = pca_instance.get_eigen_value_vector(cov_matriz)
    eigen_vec = pca_instance.get_eigenvecs(eigenvalues,eigenvectors,1)
    pca_instance.normalize()
    new_dataset = pca_instance.change_base(eigen_vec, pca_instance.normalize_data)
    print(new_dataset)
    accuracy_pca = pca_instance.knn(new_dataset)
    accuracy_without_pca = pca_instance.knn(data)
    print("Acurracy with PCA:%.3f " % np.mean(accuracy_pca))
    print("Acurracy without PCA:%.3f " % np.mean(accuracy_without_pca))


if __name__ == '__main__':
	main()