import pandas as pd
from scipy.io import arff
from sklearn import preprocessing


def readBase():
    data = arff.loadarff('DataBases\CM1.arff')
    df = pd.DataFrame(data[0])
    numberOfColumns = len(df.columns)
    justClass = df[['defects']].values     #only the classes
    justData = df.drop(['defects'], axis=1)    #Remove the last column for normalization
    justaDataValues = justData.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(justaDataValues)
    df = pd.DataFrame(x_scaled)
    df.insert(loc=numberOfColumns-1, column='defects', value=justClass)
    return df
    

            
    
        