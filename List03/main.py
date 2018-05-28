import pca
import lda
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def main():
    data_base1 = 'List03\Databases\KC1.csv'
    data_base2 = 'List03\Databases\CM1.csv'
    columns_names = "loc,v(g),ev(g),iv(g),n,v,l,d,i,e,b,t,lOCode,lOComment,lOBlank,locCodeAndComment,uniq_Op,uniq_Opnd,total_Op,total_Opnd,branchCount,defects".split(',')
    df = pd.read_csv(data_base1, names = columns_names)       #Change daba_base1 or 2
    data = df.iloc[:,:-1].copy()                              #Data without target
    target = df['defects']                                    #Target
    class_values = df['defects'].unique()                     #Number of Classes
    k_components = 3      #[1,3,5,9,15,20]                    #Components for PCA
    #PCA, LDA instances
    pca_instance = pca.pca(data,target)
    lda_instance = lda.lda(df, target,class_values)

    #PCA----------------------------------------------------------------------
    cov_matriz = pca_instance.cov_matriz()
    eigenvalues, eigenvectors = pca_instance.get_eigen_value_vector(cov_matriz)
    eigen_vec = pca_instance.get_eigenvecs(eigenvalues,eigenvectors,k_components)
    pca_instance.normalize()
    new_dataset = pca_instance.change_base(eigen_vec, pca_instance.normalize_data)

    #LDA---------------------------------------------------------------------
    mean_vectors = lda_instance.calc_mean_vect()
    data_class = lda_instance.get_data_per_class()
    s_w = lda_instance.calc_sw(mean_vectors,data_class)
    s_b = lda_instance.calc_sb(mean_vectors)
    eig_pairs = lda_instance.get_eigs(s_w, s_b)
    lda_components = lda_instance.get_k_eigenvcs(eig_pairs, len(class_values) - 1)
    new_space = pd.DataFrame(lda_instance.transform(lda_components))

    skf = StratifiedKFold(n_splits=3) #Number of folds
    knns = [1,3,5]
    print("Components PCA :%.1d" % k_components)
    for j in knns:
        print("PCA")
        print("KNN = %.1d" % j)
        accuracy_pca = pca_instance.knn(new_dataset,j,skf)
        accuracy_without_pca = pca_instance.knn(data,j,skf)
        print("Acurracy with PCA:%.3f " % np.mean(accuracy_pca))
        print("Acurracy without PCA:%.3f\n" % np.mean(accuracy_without_pca))

        print("LDA")
        accuracy_lda = lda_instance.knn(new_space,j,skf)
        accuracy_without_lda = lda_instance.knn(data,j,skf)
        print("Acurracy with LDA:%.3f " % np.mean(accuracy_lda))
        print("Acurracy without LDA:%.3f\n" % np.mean(accuracy_without_lda))


if __name__ == '__main__':
    main()